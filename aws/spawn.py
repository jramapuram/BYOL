import os
import uuid
import git
import pprint
import tarfile
import time
import boto3
import argparse
import paramiko
import numpy as np

from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='AWS Spawner')

# Node config
parser.add_argument('--uid', type=str, default=None,
                    help="tag for instance(s) (default: None)")
parser.add_argument('--instance-type', type=str, default="p3.2xlarge",
                    help="instance type (default: p3.2xlarge)")
parser.add_argument('--number-of-instances', type=int, default=1,
                    help="number of instances to spawn (default: 1)")
parser.add_argument('--storage-size', type=int, default=None,
                    help="storage in GiB (default: None)")
parser.add_argument('--storage-iops', type=int, default=None,
                    help="storage IOPS, max of 5000 (default: None)")
parser.add_argument('--keypair', type=str, default="aws",
                    help="keypair name (default: aws)")
parser.add_argument('--security-group', type=str, default="default",
                    help="security group (default: default)")
# ami-02a1a08e821457570 = nvidia-volta base
# ami-05ca11a79fafd6ebe = ubuntu18-deeplearning base
parser.add_argument('--ami', type=str, default="ami-05ca11a79fafd6ebe",
                    help="instance AMI (default: ami-05ca11a79fafd6ebe)")
parser.add_argument('--instance-region', type=str, default="us-east-1",
                    help="instance region (default: us-east-1)")
parser.add_argument('--instance-zone', type=str, default=None,
                    help="instance zone, uses cheapest by default (default: None)")
parser.add_argument('--s3-iam', type=str, default='s3-access',
                    help="iam role for s3 (default: s3-iam)")
parser.add_argument('--upper-bound-spot-multiplier', type=float, default=0,
                    help="sets the upper bound for a spot instance price as a fraction \
                    from the cheapest spot price, eg: 1.3; 0 disables and requests on-demand (default 0)")

# Command config
parser.add_argument('--no-terminate', action='store_true', default=False,
                    help='do not terminate instance after running command (default=False)')
parser.add_argument('--no-background', action='store_true', default=False,
                    help='do not run command in background (default=False)')
parser.add_argument('--cmd', type=str, default=None,
                    help="[required] run this command in the instance (default: None)")
parser.add_argument('--log-bucket', type=str, default="s3://jramapuram-logs",
                    help="[required] log to this bucket (default: s3://jramapuram-logs)")
args = parser.parse_args()


def get_all_availability_zones(client):
    '''helper to list all available zones in a region'''
    all_zones_list = client.describe_availability_zones()['AvailabilityZones']
    return [zone['ZoneName'] for zone in all_zones_list]


def get_cheapest_price(args):
    ''' gets the cheapest current AWS spot price, returns price and zone'''
    client = boto3.client('ec2', region_name=args.instance_region)
    price_dict = client.describe_spot_price_history(
        InstanceTypes=[args.instance_type],
        MaxResults=8,
        ProductDescriptions=['Linux/UNIX (Amazon VPC)']# ,
    )

    # iterate through and get the cheapest zone
    cheapest_price = np.inf
    cheapest_zone = ""
    for price in price_dict['SpotPriceHistory']:
        cheapest_price_i = float(price['SpotPrice'])
        cheapest_zone_i = price['AvailabilityZone']

        if cheapest_price_i < cheapest_price:
            cheapest_price = cheapest_price_i
            cheapest_zone = cheapest_zone_i

    # sanity checks
    assert cheapest_price is not None and cheapest_price < np.inf, "cheapest price was not determined"
    assert cheapest_zone is not None and cheapest_zone != "", "could not find cheapest zone"

    # return price and zone
    print("found cheapest price of {}$ in zone {}".format(cheapest_price, cheapest_zone))
    return cheapest_price, cheapest_zone


def attach_tag(instance_list, tag=None):
    if tag is not None:
        client = boto3.client('ec2', region_name=args.instance_region)
        client.create_tags(Resources=instance_list,
                           Tags=[{'Key':'name', 'Value':tag}])


def get_git_root():
    ''' helper to get the root git dir'''
    git_repo = git.Repo(".", search_parent_directories=True)
    return git_repo.git.rev_parse("--show-toplevel")


def tar_current_project():
    ''' helper to zip the root of this git-dir'''
    output_filename = os.path.join("/tmp", "project_{}.tar.gz".format(uuid.uuid4().hex))
    print("compressing current git repo for deployment...", end='', flush=True)
    source_dir = get_git_root()
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    print("done!")
    return output_filename

def instances_to_ips(instance_list):
    assert isinstance(instance_list, list), "need a list as input"

    # create ec2 client
    client = boto3.client('ec2', region_name=args.instance_region)

    # check if they have all spun-up; give it ~2 min
    instances_running = np.array([False for _ in range(len(instance_list))])
    total_waited_time, wait_interval = 0, 5
    max_time_to_wait = 120 / wait_interval # 2 min = 120s / [5s interval] = 24 counts

    # simple loop to wait for instances to get created
    print("waiting for instance spinup.", end='', flush=True)
    while not instances_running.all():
        reservations = client.describe_instances(InstanceIds=instance_list)['Reservations'][0]
        for i, instance in enumerate(reservations['Instances']):
            if instance['State']['Name'] == 'running':
                instances_running[i] = True

            if total_waited_time > max_time_to_wait :
                raise Exception("max time expired for instance creation".format(max_time_to_wait))

            print(".", end='', flush=True)
            time.sleep(wait_interval)
            total_waited_time += 1

    # XXX: add an extra sleep here to get ssh up
    time.sleep(40)
    print("{} instances successfully in running state.".format(len(instance_list)))

    reservations = client.describe_instances(InstanceIds=instance_list)['Reservations'][0]
    return [instance['PublicIpAddress'] for instance in reservations['Instances']]

def get_launch_spec(args, custom_zone=None):
    # custom creation of storage dict
    storage_dict = {
        'DeviceName': '/dev/sda1',
        'Ebs':{
            'DeleteOnTermination': True,
            'VolumeType': 'io1',
            'Iops': args.storage_iops,
            # 'Iops': 5000, # fast version
            # 'VolumeType': 'gp2',
            'VolumeSize': args.storage_size
        }
    } if args.storage_size is not None else {}

    # create the launch spec dict
    launch_spec_dict = {
        'SecurityGroups': [args.security_group],
        'ImageId': args.ami,
        'KeyName': args.keypair,
        'InstanceType': args.instance_type,
        'IamInstanceProfile': {
            'Name': args.s3_iam
        }
    }

    if storage_dict:
        launch_spec_dict['BlockDeviceMappings'] = [storage_dict]

    if custom_zone is not None:
        launch_spec_dict['Placement'] = {
            'AvailabilityZone': custom_zone
        }

    return launch_spec_dict


def create_spot(args):
    try:
        client = boto3.client('ec2', region_name=args.instance_region)
        cheapest_price, cheapest_zone = get_cheapest_price(args)
        max_price = cheapest_price * args.upper_bound_spot_multiplier
        print("setting max price to {}".format(max_price))

        zone_override = cheapest_zone if args.instance_zone is None else args.instance_zone
        zone_override = None if args.instance_zone == 'none' else zone_override
        #zone_override = None if args.instance_zone is None else args.instance_zone
        launch_spec_dict = get_launch_spec(args, zone_override)

        # request the node(s)
        instance_request = client.request_spot_instances(
            SpotPrice=str(max_price),
            Type='one-time',
            InstanceInterruptionBehavior='terminate',
            InstanceCount=args.number_of_instances,
            LaunchSpecification=launch_spec_dict
        )
        time.sleep(10) # XXX: sometimes remote doesn't update fast enough
        #print("\ninstance_request = {}\n".format(instance_request))

        # return the requested instances
        instance_request_ids = [ir['SpotInstanceRequestId'] for ir in instance_request['SpotInstanceRequests']]
        print("spot-request-ids: ", instance_request_ids)

        # wait till full-fulled, upto 10 min
        instances, all_instances_created = [], False
        total_waited_time, wait_interval = 0, 5
        max_time_to_wait = 600 / wait_interval # 10 min = 600s / [5s interval] = 120 counts
        print("creating {} instances, please be patient.".format(args.number_of_instances), end='', flush=True)
        #print("creating instance, please be patient.", end='', flush=True)

        while not all_instances_created:
            spot_req_response = client.describe_spot_instance_requests(
                SpotInstanceRequestIds=instance_request_ids
            )
            #print("\nspot_req = {}\n".format(spot_req_response))
            for spot_req in spot_req_response['SpotInstanceRequests']:
                if spot_req['State'] == "failed":
                    raise Exception("spot request failed:", spot_req)

                print(".", end='', flush=True)
                if 'InstanceId' not in spot_req or not spot_req['InstanceId']:
                    if total_waited_time > max_time_to_wait :
                        raise Exception("max time expired; instance creation failed".format(max_time_to_wait))

                    # sleep and increment wait count
                    time.sleep(wait_interval)
                    total_waited_time += 1
                    break

                # add the instance to the list of created instances
                instance_id = spot_req['InstanceId']
                instances.append(instance_id)
                if len(instances) == 1:
                    print("successfully created {} instance".format(len(instances)))
                    all_instances_created = True

        attach_tag(instances, tag=args.uid)
        return instances_to_ips(instances)

    except BaseException as exe:
        print(exe)


def create_ondemand(args):
    try:
        #ec2 = boto3.resource('ec2', region_name=args.instance_zone)
        ec2 = boto3.resource('ec2', region_name=args.instance_region)
        #subnet = ec2.Subnet(args.subnet) if args.subnet is not None else ec2.Subnet()
        instances = ec2.create_instances(
            MaxCount=args.number_of_instances,
            MinCount=args.number_of_instances,
            InstanceInitiatedShutdownBehavior='terminate',
            **get_launch_spec(args)
        )

        # XXX: allow server to get push request
        time.sleep(10)

        # return the requested instances ip addresses
        instance_ids = [i.id for i in instances]
        attach_tag(instance_ids, tag=args.uid)
        return instances_to_ips(instance_ids)

    except BaseException as exe:
        print(exe)


def put_file(client, local_path, remote_path):
    ''' helper to copy a local file to a remote path'''
    ftp_client = client.open_sftp()
    ftp_client.put(local_path, remote_path)
    ftp_client.close()


def run_command(cmd, hostname, pem_file, username='ubuntu',
                background=True, terminate_on_completion=True):
    key = paramiko.RSAKey.from_private_key_file(pem_file)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # format the commands to use, the order of ops is:
    # tmux_setup --> setup.sh --> yourfile.sh --> log -->shutdown
    tmux_cmd = "curl -L https://tinyurl.com/y8osnam8 -o ~/.tmux.conf"
    setup_cmd = "sh /tmp/setup.sh"
    shutdown_cmd = "sudo shutdown -P now" if terminate_on_completion else ""
    log_cmd = "aws s3 cp ~/setup.log {}/{}_setup.log ;\
    aws s3 cp ~/cmd.log {}/{}_cmd.log".format(
        args.log_bucket, hostname.replace(".", "_"),
        args.log_bucket, hostname.replace(".", "_")
    )

    # build the final command to send over ssh
    orig_cmd = cmd # cache for use in file transfer
    cmd = "{} ; tmux new-session -d -s runtime; \
    tmux send-keys \"{} &> ~/setup.log ; sh /tmp/{} &> ~/cmd.log ; {} ; {} \" C-m ; \
    tmux detach -s runtime".format(
        tmux_cmd, setup_cmd, cmd, log_cmd, shutdown_cmd
    )

    # setup client
    client.connect(hostname=hostname, username=username, pkey=key)

    # tar the current project
    project_filename = tar_current_project()

    # send all files from current directory to remote server
    all_files = [local_file for local_file in os.listdir(".")
                 if local_file == 'setup.sh'
                 or local_file == orig_cmd] + [project_filename]
    print("transferring {} to host {}".format(all_files, hostname))
    for local_file in all_files:
        remote_file = os.path.join("/tmp", os.path.basename(local_file))
        put_file(client, local_file, remote_file)

    # remove the compressed project to not cause clutter
    os.remove(project_filename)

    # run async
    if background:
        print("running {} asynchronously".format(cmd))
        transport = client.get_transport()
        channel = transport.open_session()
        channel.exec_command(cmd)
        return {
            'stdout': 'running in background',
            'stderr': ''
        }

    # run synchronously
    print("running {} synchronously".format(cmd))
    cmd = "{} && {} ; {}".format(
        setup_cmd, cmd, shutdown_cmd
    )
    stdin, stdout, stderr = client.exec_command(cmd)
    stdout = stdout.read().decode('ascii')
    stderr = stderr.read().decode('ascii')
    client.close()

    # return the text in string format
    return {
        'stdout': stdout,
        'stderr': stderr
    }


def main(args):
    # execute the config
    if args.upper_bound_spot_multiplier > 0:
        print("attempted to request spot instance w/max price = {} x cheapest".format(
            args.upper_bound_spot_multiplier)
        )
        instance_ips = create_spot(args)
    else:
        print("creating on-demand instance")
        instance_ips = create_ondemand(args)

    def run_command_on_remote(instance_ip):
        # run the specified command over ssh
        # also append shutdown command to terminate
        pem_file_path = os.path.join(os.path.expanduser('~'), ".ssh", args.keypair + ".pem")
        cli_out = run_command(args.cmd, instance_ip, pem_file_path,
                              background=not args.no_background,
                              terminate_on_completion=not args.no_terminate)
        print("[{}][stdout]: {}".format(instance_ip, cli_out['stdout']))
        print("[{}][stderr]: {}".format(instance_ip, cli_out['stderr']))

    Parallel(n_jobs=len(instance_ips))(
        delayed(run_command_on_remote)(ip) for ip in instance_ips)

if __name__ == "__main__":
    # print the config and sanity check
    pprint.PrettyPrinter(indent=4).pprint(vars(args))
    assert args.cmd is not None, "need to specify a command"
    main(args)
