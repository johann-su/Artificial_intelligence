# Training Deep Neural Nets in the cloud
For large NN it might be a good idea to use more Powerful computers than the one you have at home (except you already have a gaming PC with a realy good graphics card).<br>
__NOTE:__ Training in the cloud is not free!!! Depending on the instance type you choose you have to pay a set amout per our. Please check out the up to date pricing tables of your provider.<br>

## AWS
[Amazon Web Services](https://aws.amazon.com/de/) is the market leader in Cloud Services. They offer a wide veriaty of services, including cloud computing.<br>

### SageMaker
[SageMaker](https://aws.amazon.com/de/sagemaker/) is a tool provided by aws to build, train and deploy Neural Networks. It has a build in Jupyter Notebook editor and is very well optimized for Machine Learning. It also contains some popular problems with a solution.<br>
Check out this tutorial on how to [get started with SageMaker](https://www.youtube.com/watch?v=tBRHh_V8vjc).<br>
Here are the [prices for the instances](https://aws.amazon.com/de/sagemaker/pricing/).

### EC2 instance
While SageMaker is specificly made for Machine Learning, you can use a normal server instead as well.<br>
__NOTE:__ You may have to request a limit upgrade before you can launch an instance. Follow the instructions on the site.
<br>
[Tutorial on how to get started](https://www.youtube.com/watch?v=vfjbECWi7F0).<br>
[Pricing](https://aws.amazon.com/de/ec2/pricing/on-demand/).

## GCP
Google Cloud Plattform is a similar service to the one provided by amazon. But since Google developed Tensorflow, they have some advanteges in terms of Speed (they developed a new device called TPU (Tensor Processing Unit) to run tasks in Tensorflow even faster).

## ML Engine
AI Plattform nearly the same as amazon SageMaker, with the diffrence, that you can choose a TPU. <br>
[As always a Tuturial](https://www.youtube.com/watch?v=VxnHf-FfWKY), <br>
[and Pricing](https://cloud.google.com/ml-engine/docs/pricing).

### Normal vm
The normal virtual machine to Train ML Models. <br>
[Heres the tutorial on how to get started](https://www.youtube.com/watch?v=chk2rRjSn5o).<br>
[And Pricing](https://cloud.google.com/compute/vm-instance-pricing).

__Remember to always shut down your instance after usage to avoid hefty bills!__
