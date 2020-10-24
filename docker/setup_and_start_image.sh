docker build -t hmc_tomography_image --build-arg SSH_KEY="$(cat ~/.ssh/id_rsa)" .
# docker build -t hmc_tomography_image  .
docker run -it --rm hmc_tomography_image bash