import numpy as np
import time
import matplotlib.pyplot as plt

def distance_between_linear_function_and_neural_network(env, actor, K, terminal_err=0.01, rounds=10, steps=500):
	"""sum distance between the output of LF and NN 
		until the state's MSE*dim less than terminal_err
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	    actor (DDPG.ActorNetwork): actor
	    K (numpy.matrix): coefficient of LF
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	"""
	distance = 0
	sum_steps = 0
	temp_env_ter_err = env.terminal_err
	env.terminal_err = terminal_err

	for i in range(rounds):
		env.reset()
		ep_distance = 0
		for s in range(steps):
			xk, r, terminal = env.observation()
			if r == env.bad_reward:
				sum_steps -= s
				distance -= ep_distance
			if terminal:
				break
			sum_steps += 1
			u1 = actor.predict(np.reshape(np.array(xk), (1, actor.s_dim)))
			u2 = K.dot(xk)
			env.step(np.reshape(u1, (actor.a_dim, 1)))
			distance += np.linalg.norm(u1-u2)
			ep_distance += np.linalg.norm(u1-u2)

	env.terminal_err = temp_env_ter_err
	if sum_steps == 0:
		return 1
	return float(distance)/sum_steps


def neural_network_performance(env, actor, terminal_err=0.01, rounds=10, steps=500):
	"""Measured by the steps NN took until 
	the sum of state absolute value less than terminal_err
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	    actor (DDPG.ActorNetwork): actor
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	"""
	sum_steps = 0
	temp_env_ter_err = env.terminal_err
	env.terminal_err = terminal_err
	success_rounds = rounds

	for i in range(rounds):
		env.reset()
		for s in range(steps):
			xk, r, terminal = env.observation()
			if r == env.bad_reward:
				sum_steps -= s
				success_rounds -= 1
			if terminal:
				break
			sum_steps += 1
			u = actor.predict(np.reshape(np.array(xk), (1, actor.s_dim)))
			env.step(np.reshape(u, (actor.a_dim, 1)))

	env.terminal_err = temp_env_ter_err
	if success_rounds == 0:
		return steps+1

	return float(sum_steps)/success_rounds

def neural_network_performance_converge(env, actor, terminal_err=0.01, rounds=10, steps=500, measure_steps=10):
	"""Measured by the steps NN took until converge
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	    actor (DDPG.ActorNetwork): actor
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	    measure_step(int): measure the state change during this step
	"""
	sum_steps = 0
	success_rounds = rounds

	for i in range(rounds):
		pre_x = env.reset()
		change_sum = 0.0
		loop_sum_steps = 0
		slide_window = []

		assert steps > 0
		for s in range(steps):
			xk, r, terminal = env.observation()
			slide_window.append(np.sum(np.abs(xk - pre_x)))
			pre_x = xk
			change_sum += slide_window[-1]
			if len(slide_window) == measure_steps:
				loop_sum_steps += 1
				if change_sum < terminal_err:
					break
				change_sum -= slide_window[0]
				slide_window.pop(0)

			if r == env.bad_reward:
				sum_steps -= loop_sum_steps
				success_rounds -= 1
				break

			u = actor.predict(np.reshape(np.array(xk), (1, actor.s_dim)))
			env.step(np.reshape(u, (actor.a_dim, 1)))
		sum_steps += loop_sum_steps

	if success_rounds == 0:
		return steps+1

	return float(sum_steps)/success_rounds

def neural_network_with_shield_performance(env, shield, terminal_err=0.01, rounds=100, steps=500):
	"""Measured by the steps NN with shield took until 
	the sum of state absolute value less than terminal_err
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	   	shield (Shield.shield): shield
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	"""
	sum_steps = 0
	temp_env_ter_err = env.terminal_err
	env.terminal_err = terminal_err
	actor = shield.actor

	for i in range(rounds):
		env.reset()
		for s in range(steps):
			xk, r, terminal = env.observation()				
			if terminal:
				break

			sum_steps += 1

			u = actor.predict(np.reshape(np.array(xk), (1, actor.s_dim)))
			if shield.detactor(xk, u):
				u = shield.call_shield(xk, mute=True)

			env.step(np.reshape(u, (actor.a_dim, 1)))

	env.terminal_err = temp_env_ter_err

	return float(sum_steps)/rounds


def linear_function_performance(env, K, terminal_err=0.01, rounds=100, steps=500):
	"""Measured by the steps LF took until 
	the sum of state absolute value less than terminal_err
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	    K (numpy.matrix): coefficient of LF
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	"""
	sum_steps = 0
	temp_env_ter_err = env.terminal_err
	env.terminal_err = terminal_err
	for i in range(rounds):
		env.reset()
		for s in range(steps):
			xk, r, terminal = env.observation()
			if terminal:
				break
			sum_steps += 1
			u = K.dot(xk)
			env.step(u)

	env.terminal_err = temp_env_ter_err
	return float(sum_steps)/rounds


def linear_function_performance_converge(env, K, terminal_err=0.01, rounds=100, steps=500, measure_steps=10):
	"""Measured by the steps LF took until 
	the sum of state absolute value less than terminal_err
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	    K (numpy.matrix): coefficient of LF
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	"""
	"""Measured by the steps NN took until converge
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	    actor (DDPG.ActorNetwork): actor
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	    measure_step(int): measure the state change during this step
	"""
	sum_steps = 0
	success_rounds = rounds

	for i in range(rounds):
		pre_x = env.reset()
		change_sum = 0.0
		loop_sum_steps = 0
		slide_window = []

		assert steps > 0
		for s in range(steps):
			xk, r, terminal = env.observation()
			slide_window.append(np.sum(np.abs(xk - pre_x)))
			pre_x = xk
			change_sum += slide_window[-1]
			if len(slide_window) == measure_steps:
				loop_sum_steps += 1
				if change_sum < terminal_err:
					break
				change_sum -= slide_window[0]
				slide_window.pop(0)

			if r == env.bad_reward:
				sum_steps -= loop_sum_steps
				success_rounds -= 1
				break

			u = K.dot(xk)
			env.step(u)
		sum_steps += loop_sum_steps

	if success_rounds == 0:
		return steps+1

	return float(sum_steps)/success_rounds

def timeit(func):
	"""Record time a function runs with, print it to standard output
	
	Args:
	    func (callable): The function measured
	"""
	def wrapper(*args, **kvargs):
		start = time.time()
		ret = func(*args, **kvargs)
		end = time.time()
		t = end-start
		print func.__name__, 'run time:', t, 's'
		return ret

	return wrapper


def find_boundary(x, x_max, x_min):
    """find if x is between x_max and x_min
    if not, extending x_max and x_min with x
    
    Args:
        x (np.array): state
        x_max (np.array): state max values
        x_min (np.array): state min values
    """
    max_update = (x > x_max)
    min_update = (x < x_min)
    x_max = np.multiply(x,max_update) + np.multiply(x_max, np.logical_not(max_update))
    x_min = np.multiply(x,min_update) + np.multiply(x_min, np.logical_not(min_update))

    return x_max, x_min

def draw_K (env, K, simulation_steps, x0=None, names=None):
  if x0 is None: 
  	x0 = env.reset()
  else:
  	x0 = env.reset(x0)
  if names is None:
  	names = {i:"x"+str(i) for i in range(len(env.s_min))}

  return draw_K_helper (env, K, x0, simulation_steps, names)    

def draw_K_helper (env, K, x0, simulation_steps, names):
    time = np.linspace(0, simulation_steps, simulation_steps, endpoint=True)
    xk = x0 #np.matrix(".0 ; 0 ; .0 ; 0.1")

    XS = []
    for i in range(len(names)):
        XS.append([])
    reward = 0
    for t in time:
        uk = K.dot(xk)
        for i, k in enumerate(sorted(names.keys())):
          val = xk[k,0]
          XS[i].append(val)

        reward += env.reward(xk, uk)
        # Use discrete or continuous semantics based on user's choice
        xk, _, t = env.step(uk)
        if t: 
        	break

    print "Score of the trace: {}".format(reward) 

    for i, k in enumerate(sorted(names.keys())):
        plt.plot(time, XS[i], label=names[k])

    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    return xk


def draw_actor(env, actor, simulation_steps, x0=None, names=None):
  if x0 is None: 
  	x0 = env.reset()
  else:
  	x0 = env.reset(x0)
  if names is None:
  	names = {i:"x"+str(i) for i in range(len(env.s_min))}

  return draw_actor_helper (env, actor, x0, simulation_steps, names)    

def draw_actor_helper (env, actor, x0, simulation_steps, names):
    time = np.linspace(0, simulation_steps, simulation_steps, endpoint=True)
    xk = x0 

    XS = []
    for i in range(len(names)):
        XS.append([])
    reward = 0
    for t in time:
        uk = np.reshape(actor.predict(np.reshape(np.array(xk), (1, actor.s_dim))), (actor.a_dim, 1))
        for i, k in enumerate(sorted(names.keys())):
          val = xk[k,0]
          XS[i].append(val)

        reward += env.reward(xk, uk)
        xk, _, t = env.step(uk)
        if t:
        	break

    print "Score of the trace: {}".format(reward) 

    for i, k in enumerate(sorted(names.keys())):
        plt.plot(time, XS[i], label=names[k])


    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    return xk


