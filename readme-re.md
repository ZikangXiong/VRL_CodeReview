Artifact Evaluation for Paper #291
==================================

An Inductive Synthesis Framework for Verifiable Machine Learning
==================================


We provide a prepared docker image to run our artifact. The experimental results collected in the docker image may be different than what we reported in the paper because (1) it is a docker environment with limited memory (2) the tool in the docker image may have different behaviors, i.e., the random generator used by the TensorFlow library may be different, which may lead to a significantly different search direction of both a synthesized program and an inferred invariant. However, our artifact suffices to prove the reproducibility of our work.

Please follow the steps below to get our artifact:

# Getting Started Guide: 
1. Install Docker:  
[https://docs.docker.com/get-started/](https://docs.docker.com/get-started/)

2. Pull Verifiable Reinforcement Learning Environment docker image:   
```  
docker pull caffett/vrl_env
```
3. Start docker environment:  
```  
docker run -it caffett/vrl_env:v0 /bin/bash
```
4. Clone VRL code:  
```
git clone git@github.com:caffett/VRL_CodeReview.git
```

# Step-by-Step Instructions
## Run Command
Pretrained neural networks are provided in our artifact. We do not provide an interface to retrain neural networks because retraining typically requires significant manual efforts to adjust a number of training parameters and is often time-consuming. Our pretrained neural models is stored at `ddpg_chkp/<model_name>/<network_structure>`.

Our tool provides a python script for each of our benchmarks. Given a benchmark, the user just needs to type:

```
python <benchmark_name> 
[--nn_test | --shield_test | --retrain_shield | --test_episodes=TEST_EPISODES]
``` 

There are 4 flags: 
**--nn\_test**: adding this flag runs the pretrained neural network controller alone without shield protection.  
**--shield\_test**:  adding this flag runs the pretrained neural network controller in tandem with a pre-synthesized and verified program distilled from the network to provide shield protection.  
**--retrain\_shield**: adding this flag re-synthesizes a deterministic program to be used for shielding.
**--test\_episodes**: This parameter specifies the number of simulation steps used for each simulation run. The default value is 100.    

## Getting Results
### Run a Single Benchmark
After running a benchmark, our tool reports the total simulation time and the number of times that the system enters an unsafe state.  
For example, running `python 4-car-platoon.py --nn_test --test_episodes=1000` may produce the following result:

<center>
![](https://user-images.githubusercontent.com/11462215/53280122-bcd48b00-36e4-11e9-83aa-fa171fe74e7c.png)
![](https://user-images.githubusercontent.com/11462215/53280155-18067d80-36e5-11e9-9a9f-3a767f0b12f3.png)
</center>

The system is indeed unsafe since a number of safety violations were observed.

Running a neural network controller in tandem with a verified program distilled from it can eliminate all those unsafe neural actions. Our tool produces in its output the number of interventions from the program to the neural network controller. It also gives the total running time including the additional cost of shielding.

Running `python 4-car-platoon.py --shield_test --test_episodes=1000` may produce the following result (using a pre-synthesized and verified program):
<center>
![image](https://user-images.githubusercontent.com/11462215/53280233-d1fde980-36e5-11e9-8e7a-82111927ad56.png)
</center>

Based on a neural network's simulation time without and with running a shield, we can calculate the overhead of using the shield.  

```
Overhead = (shield_test_runing_time - neural_network_test_runing_time) /
neural_network_test_runing_time * 100%
```
  
For each benchmark, with the protection of a shield, our system never enters an unsafe state. We may get the following result for all our benchmarks.

<center>
![image](https://user-images.githubusercontent.com/11462215/53280265-21dcb080-36e6-11e9-9ed7-1a9146b6529e.png)
</center>

Running with --retrain_shield can re-synthesize a new deterministic program to replace the original one. After re-synthesis, our tool produces the total synthesis time.  For example, we may get the following result by running `python 4-car-platoon.py --retrain_shield --test_episodes=1000`. 
<center>
![image](https://user-images.githubusercontent.com/11462215/53280299-6f591d80-36e6-11e9-88d3-0b83c97dec26.png)
</center>

We count how many iterations our algorithm needs to synthesize a deterministic program and this result corresponds to the size of a synthesized program (i.e., the number of conditional branches in the synthesized program). 
<center>
![image](https://user-images.githubusercontent.com/11462215/53280317-a5969d00-36e6-11e9-86d7-0a13f31b1c57.png)
</center>

### Run All Benchmarks

We also provide some scripts to run all of our benchmarks in a batch mode:

`./run_test_100ep.sh`: Run all the benchmarks with pretrained neural networks. The number of steps used in each simulation run is set to 100.     
`./run_test_1000ep.sh`: Run all the benchmarks with pretrained neural networks. The number of steps used in each simulation run is set 1000.    
`./run_retrain_shield.sh`: Retrain deterministic programs for all the benchmarks.
