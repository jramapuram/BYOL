# AWS Launcher

Creates AWS instances for ML jobs

## Spot - Operations
The spot feature of this code does the following:

  - *Spot pricing and requester* : creates spot instances by 1) finding cheapest price & 2) setting upper bound spot price cheapest * `upper-bound-spot-multiplier` (from argparse)

  - *Wait for spot request to be accepted* : once a request is submitted it needs to be accepted. The program currently just sits and waits till this is done. There is a timeout on this (currently hardcoded) to exit the operation if it fails. If it does fail check your EC2 console --> probably a node config was not available.

  - *Wait for instance to be created and get ip*: once the previous step is complete the program waits for the instances to be moved into `running` state and then grabs the ipv4 address(es)

  - *SCP aftering tar-ing `..` to /tmp/project-\<UUID\>.tar.gz : self explanatory

  - *Run command order* : 1) curl .tmux.conf from my dotfiles repo to ~/.tmux.conf ; 2) run `setup.sh` in current directory ; 3) run whatever you pass to `--cmd`, eg: an .sh file. 4) push logs (for setup.sh and cmd) to an s3 bucket where the files are stored as \<ip\>_cmd.log & \<ip\>_setup.log 5) calls instance shutdown (which forces termination and deletion of the drive)


## Example calls

The following will train a fashion VAE on an AWS spot instance.

``` bash
➜ python spawn.py --instance-type=p3.2xlarge --number-of-instances=1 --upper-bound-spot-multiplier=1.2 --cmd=run_test.sh
{   'ami': 'ami-0af8dc9d28a9aed78',
    'cmd': 'run_test.sh',
    'instance_region': 'us-east-1',
    'instance_type': 'p3.2xlarge',
    'instance_zone': None,
    'keypair': 'aws',
    'log_bucket': 's3://jramapuram-logs',
    'no_background': False,
    'no_terminate': False,
    'number_of_instances': 1,
    's3_iam': 's3-access',
    'security_group': 'default',
    'storage_size': 150,
    'upper_bound_spot_multiplier': 1.2}
attempted to request spot instance w/max price = 1.2 x cheapest
found cheapest price of 1.2166$ in zone us-east-1c
setting max price to 1.45
spot-request-ids:  ['YOUR_REQUEST_ID']
creating 1 instances, please be patient..successfully created 1 instances
waiting for instance spinup....1 instances successfully in running state.

running curl -L https://tinyurl.com/y8osnam8 -o ~/.tmux.conf ; tmux new-session -d -s runtime;     tmux send-keys "sh /tmp/setup.sh > ~/setup.log ; sh /tmp/two_digit_clutter_id_exp1.sh > ~/cmd.log ; aws s3 cp ~/setup.log s3://jramapuram-logs/34_237_138_133_setup.log ;    aws s3 cp ~/cmd.log s3://jramapuram-logs/34_237_138_133_cmd.log ; sudo shutdown -P now " C-m ;     tmux detach -s runtime asynchronously
[XXX.XXX.XXX.XXX][stdout]: running in background
[XXX.XXX.XXX.XXX][stderr]:

```
where `XXX.XXX.XXX.XXX` is the node IP address and `YOUR_REQUEST_ID` is the spot-request ID.
