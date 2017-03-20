wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
bash Anaconda2-4.3.1-Linux-x86_64.sh
export PATH="/home/ubuntu/anaconda2/bin:$PATH"
conda install pytorch torchvision -c soumith
conda install opencv
conda install accelerate
mkdir upload
chmod 0777 upload
cd upload
export PATH="/home/ubuntu/anaconda2/bin:$PATH"
pip install captcha
mkdir data
