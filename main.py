import pygame, random, sys
from pygame.locals import *
import cv2
from collections import deque
import tensorflow as tf
import numpy as np
import NetworkStructure1
import heapq
import random

def collide(x1, x2, y1, y2, w1, w2, h1, h2):
	if x1+w1>x2 and x1<x2+w2 and y1+h1>y2 and y1<y2+h2:
		return True
	else:
		return False

def initialise() :
	global xs,lastState
	global ys,dirs,score,applepos
	global lastAction
	global numberGames
	numberGames += 1
	x1 = random.randint(25,45)
	y1 = random.randint(25,45)
	xs = [x1, x1, x1, x1, x1];
	ys = [y1+20, y1+15, y1+10, y1+5, y1];

	for i in range(len(lastAction)) :
		lastAction[i] = 0
	if numberGames%2 == 0 :	
		lastAction[0] = 1
		dirs = 0		
	else :
		lastAction[2] = 1
		dirs = 2
		ys = [y1, y1+5, y1+10, y1+15, y1+20];
	score = 0;
	lastState = None
	applepos = (random.randint(5, 70), random.randint(5, 70));
	if numberGames<= 10000 and numberGames%2 == 1:
		applepos = (random.randint(20, 60), random.randint(20, 60))

def CheckTerminal(xs,ys) :
	i = len(xs)-1
	while i >= 2:
		if collide(xs[0], xs[i], ys[0], ys[i], 5, 5, 5, 5):
			break
		i-= 1
	if i>=2 :
		return True
	if xs[0] < 0 or xs[0] > 75 or ys[0] < 0 or ys[0] > 75:
		return True
	return False

def randomiseAction(numberActions) :
	new_action = np.zeros([numberActions])
	action_index = random.randrange(numberActions)
	new_action[action_index] = 1
	return new_action	

def getRandomAction() :
	global numberActions
	global dirs
	Actions = randomiseAction(numberActions)
	act_index = np.argmax(Actions)
	if dirs == 0 :
		while(act_index == 2) :
			Actions = randomiseAction(numberActions)
			act_index = np.argmax(Actions)
	elif dirs == 2 :
		while(act_index == 0) :
			Actions = randomiseAction(numberActions)
			act_index = np.argmax(Actions)	
	elif dirs ==3 :
		while(act_index == 1) :
			Actions = randomiseAction(numberActions)
			act_index = np.argmax(Actions)	
	elif dirs == 1 :
		while(act_index == 3) :
			Actions = randomiseAction(numberActions)
			act_index = np.argmax(Actions)

	return Actions

def getNetworkAction() :
	
	global numberActions
	global dirs
	global inpLayer,outLayer,lastState,sess
	global net1
	
	max1,max2 = getActions(net1,lastState,sess)
	maxIndex = max1
	if dirs == 0 and max1 == 2 :
		maxIndex = dirs
	if dirs == 2 and max1 == 0 :
		maxIndex = dirs
	if dirs == 3 and max1 == 1 :
		maxIndex = dirs
	if dirs == 1 and max1 == 3 :
		maxIndex = dirs
	new_action = np.zeros([numberActions])
	new_action[maxIndex] = 1
	return new_action						

def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars/2]):
		 op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)

def choose_next_action() :
	global observations,random_prob,FINAL_RANDOM_ACTION_PROB,INITIAL_RANDOM_ACTION_PROB
	global OBSERVATION_STEPS,EXPLORE_STEPS
	if random.random() <= random_prob :
		lastAction = getRandomAction() 
	else :
		lastAction = getNetworkAction()

	if len(observations) > OBSERVATION_STEPS and random_prob > FINAL_RANDOM_ACTION_PROB :
		random_prob -= (INITIAL_RANDOM_ACTION_PROB - FINAL_RANDOM_ACTION_PROB) / EXPLORE_STEPS

		
	return lastAction

def train(sess,net1,net2,observations,observations1,batchSize,FUTURE_REWARD_DISCOUNT) :
	global random_prob
	if random_prob >= 0.01 :
		mini_batch = random.sample(observations, batchSize-15)
		mini_batch1	= random.sample(observations1, 15)
		for i in range(15) :
			mini_batch.append(mini_batch1[i])
	else :		
		mini_batch = random.sample(observations, batchSize)
	previous_states = [d[0] for d in mini_batch]
	actions = [d[1] for d in mini_batch]
	rewards = [d[2] for d in mini_batch]
	current_states = [d[3] for d in mini_batch]
	agents_expected_reward = []
	expected_pos = sess.run(net1.predict_action, feed_dict={net1.input_layer: current_states})
	
	agents_reward_per_action = sess.run(net2.output_layer, feed_dict={net2.input_layer: current_states})

	for i in range(len(mini_batch)):
		if mini_batch[i][4]:
		    agents_expected_reward.append(rewards[i])
		else:
		    agents_expected_reward.append(
		        rewards[i] + FUTURE_REWARD_DISCOUNT * agents_reward_per_action[i][expected_pos[i]])	    
	_,loss = sess.run([net1.train_operation,net1.cost],feed_dict={net1.input_layer: previous_states,net1.feedAction: actions,net1.target:agents_expected_reward})

	return loss
def getActions(net1,feedInp,sess) :
	x,y,z = feedInp.shape
	feedInp = np.reshape(feedInp,(1,x,y,z))
	out1 = sess.run(net1.output_layer,feed_dict={net1.input_layer:feedInp})[0]
	maxArr = heapq.nlargest(2,range(len(out1)), key=out1.__getitem__)
	return maxArr[0],maxArr[1]


numberActions = 4
totalFrames = 4
learningRate = 1e-4
maxMemory = 100000
RESIZED_SCREEN_X = 80
RESIZED_SCREEN_Y = 80
INITIAL_RANDOM_ACTION_PROB = 0.9
FINAL_RANDOM_ACTION_PROB = 0.05
OBSERVATION_STEPS = 20000.
EXPLORE_STEPS = 100000.

checkpoint_path = "Models3"
os.mkdir(checkpoint_path)

checkpoint_save_steps = 4000
batchSize = 100
TARGET_UPDATION_STEP = 1000
LOSS_DISPLAY = 100
FUTURE_REWARD_DISCOUNT = 0.9
catch_Display = 2000

xs = [45, 45, 45, 45, 45];
ys = [50, 45, 40, 35, 30];
dirs = 0;
score = 0;
numberTrain = 0
numberGames = 0
applepos = (random.randint(0, 75), random.randint(0, 75));
pygame.init();
s=pygame.display.set_mode((80, 80));
pygame.display.set_caption('Snake');
appleimage = pygame.Surface((5, 5));
appleimage.fill((255, 255, 255));
img = pygame.Surface((5, 5));
img.fill((11, 220, 0));
f = pygame.font.SysFont('Arial', 20);
clock = pygame.time.Clock()

observations = deque()
observations1 = deque()
lastState = None
lastAction = np.zeros([numberActions])
lastAction[0] = 1
totalActions = 0
random_prob = INITIAL_RANDOM_ACTION_PROB

tau = 1.0
tf.reset_default_graph()
net1 = NetworkStructure1.create_network(numberActions,learningRate)
targetNet = NetworkStructure1.create_network(numberActions,learningRate)

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)
targetOps1 = updateTargetGraph(trainables,1.0)

sess = tf.Session()

saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())
updateTarget(targetOps1,sess)


checkpoint = tf.train.get_checkpoint_state(checkpoint_path)


if checkpoint and checkpoint.model_checkpoint_path:
	saver.restore(sess, checkpoint.model_checkpoint_path)
	print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)

previousCatch = 0
catchDifference = 0
totalCatch = 0
overallCatch = 0

while True :
	terminal = False
	reward = -1
	clock.tick(10)
	totalActions+=1
	
	gameArray = pygame.surfarray.array3d(pygame.display.get_surface())
	gameArrBinary = cv2.cvtColor(cv2.resize(gameArray,(RESIZED_SCREEN_X, RESIZED_SCREEN_Y)),cv2.COLOR_BGR2GRAY)
	#_, gameArrBinary = cv2.threshold(gameArrModified, 1, 255, cv2.THRESH_BINARY)

	for e in pygame.event.get():
		if e.type == QUIT:
			sys.exit(0)
	#pygame.time.wait(0);
	
	if collide(xs[0], applepos[0], ys[0], applepos[1], 5, 5, 5, 5):
		score+=1;
		reward = 50
		xs.append(700);
		ys.append(700);
		
		applepos=(random.randint(5,70),random.randint(5,70))
		catchDifference = totalActions - previousCatch
		previousCatch = totalActions
		totalCatch+= 1
		overallCatch+= 1
	if CheckTerminal(xs,ys) :
		terminal = True
		reward = -10
		
	if lastState is None :
		lastState = np.stack(tuple(gameArrBinary for _ in range(totalFrames)), axis=2)
		
	else :
		#print 'hi'
		screen_resized_binary = np.reshape(gameArrBinary,(RESIZED_SCREEN_X, RESIZED_SCREEN_Y, 1))
		current_state = np.append(lastState[:, :, 1:], screen_resized_binary, axis=2)
		if reward == 50 :
			observations1.append((lastState,lastAction,reward,current_state,terminal))	
		observations.append((lastState,lastAction,reward,current_state,terminal))
		lastState = current_state
		lastAction = choose_next_action()
		dirs = np.argmax(lastAction)
		
	if len(observations) > maxMemory :
		observations.popleft()	

	if totalActions % catch_Display == 0 :
		print('Food catch rate is %d and overall catches are %d'%(totalCatch,overallCatch))
		
		totalCatch = 0

	if len(observations) > OBSERVATION_STEPS :
		numberTrain += 1
		loss = train(sess,net1,targetNet,observations,observations1,batchSize,FUTURE_REWARD_DISCOUNT)
		if numberTrain% TARGET_UPDATION_STEP == 0 :
			updateTarget(targetOps,sess)
		if numberTrain% LOSS_DISPLAY == 0 or numberTrain == 1:
			print('the loss is %g'%loss)
			
		if numberTrain % checkpoint_save_steps == 0 :
			saver.save(sess,checkpoint_path+'/network',global_step = numberTrain)
			print 'model saved'

			
	if terminal == True :
		initialise()
		continue	
	i = len(xs)-1
	while i >= 1:
		xs[i] = xs[i-1];
		ys[i] = ys[i-1];
		i -= 1
	if dirs==0:
		ys[0] += 5
	elif dirs==1:
		xs[0] += 5
	elif dirs==2:
		ys[0] -= 5
	elif dirs==3:
		xs[0] -= 5
	s.fill((0, 0, 0))	
	for i in range(0, len(xs)):
		s.blit(img, (xs[i], ys[i]))
	s.blit(appleimage, applepos);
	#t=f.render(str(score), True, (0, 0, 0));
	#s.blit(t, (10, 10));
	pygame.display.update()
					
					
			


