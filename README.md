# For inference of images in a private S3 bucket

aws_access_key_id,aws_secret_access_key, and aws_session_token are in https://labs.vocareum.com/main<br>

input command as:<br>
bash setup.sh
python3 inference2.py aws_access_key_id aws_secret_access_key aws_session_token<br>

## note
Images cannot be showned/ploted in CLI of EC2.<br>
Test/statistic is done in Sagemaker.