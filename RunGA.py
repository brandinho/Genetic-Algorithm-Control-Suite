from dm_control import suite
import numpy as np
from PIL import Image
import subprocess
import cv2
import os
import re

import tensorflow as tf
from PolicyNetwork import Actor
from GeneticAlgorithm import genetic_algorithm

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


total_generations = 1000
population_size = 40
crossover_probability = 1
mutation_probability = 0.2
rounds_per_generation = 1 # We might choose to run it multiple times per generation because of random starting points

# Load your environment for each of the agents in the population
# We are loading a list of environments with the same seed so that we can actually compare the agents on an apple-to-apple basis
# at each generation
envs = [suite.load(domain_name="cheetah", task_name="run", task_kwargs={'random': 99}) for _ in range(population_size)]

# Step through an episode and print out reward, discount and observation.
action_spec = envs[0].action_spec()
time_steps = [x.reset() for x in envs]


# Reset frames folder to create videos during training
subprocess.call([ 'rm', '-rf', 'framesStart' ])
subprocess.call([ 'mkdir', '-p', 'framesStart' ])

subprocess.call([ 'rm', '-rf', 'framesMiddle' ])
subprocess.call([ 'mkdir', '-p', 'framesMiddle' ])

subprocess.call([ 'rm', '-rf', 'framesEnd' ])
subprocess.call([ 'mkdir', '-p', 'framesEnd' ])

height = 480
width = 480

def flatten_input(time_step_input):
    flattened_input = []
    for x in time_step_input.observation.values():
        if x.shape == ():
            flattened_input.extend(list([x]))
        else:
            flattened_input.extend(list(x))
    return flattened_input

flattened_input = flatten_input(time_steps[0])
input_dimension = len(flattened_input)

# Initialize the model
tf.reset_default_graph()
sess = tf.Session()
agent = Actor(sess, input_dimension, action_spec.minimum.shape[0], population_size)
sess.run(tf.global_variables_initializer())

neural_net_dim = [sess.run(v).shape for v in tf.trainable_variables()]

parameter_shapes = []
total_parameters = 0
for i in range(len(neural_net_dim)):
    multiplier = 1
    for j in range(len(neural_net_dim[i])):
        multiplier *= neural_net_dim[i][j]
    parameter_shapes.append(multiplier)
    total_parameters += multiplier

generation_counter = 0
agent_index = 0
time_step_counter = 0
generation_round = 1

agent.update_parameters(agent.population[agent_index,], parameter_shapes, neural_net_dim)

max_fitness = np.zeros(total_generations)
rewards = []
current_fitness = []

# Start the genetic algorithm training process
while generation_counter < total_generations:
    if time_step_counter >= 500:
        if generation_round == 1:
            current_fitness.append(sum(rewards))
        else:
            current_fitness[-1] += sum(rewards)
        rewards = []
        time_steps[agent_index] = envs[agent_index].reset()
        time_step_counter = 0        
        if generation_round == rounds_per_generation:
            if agent_index < population_size - 1:
                agent_index += 1
                agent.update_parameters(agent.population[agent_index,], parameter_shapes, neural_net_dim)  
                generation_round = 1
            else:
                generation_round += 1
        else:
            generation_round += 1

    if len(current_fitness) == population_size and generation_round == rounds_per_generation + 1:
        fitness_array = np.array(current_fitness)
        elite_population = agent.population[int(fitness_array.argsort()[-1:][::-1]),]
        max_fitness[generation_counter] = max(fitness_array)
        agent = genetic_algorithm(agent, fitness_array, population_size, "Crossmutate", crossover_probability = crossover_probability, 
                                  crossover_type = "One Point", random_type = True, mutate_probability = mutation_probability, 
                                  noise_scale = 0.1, selection_method = "Roulette")
        if generation_counter % 1 == 0:
            print("Generation: {}, Fitness: {}".format(generation_counter, max_fitness[generation_counter]))
            
        current_fitness = []
        agent_index = 0
        agent.update_parameters(agent.population[agent_index,], parameter_shapes, neural_net_dim)
        generation_counter += 1
        generation_round = 1
    
    model_input = np.array(flatten_input(time_steps[agent_index])); model_input[2] /= 10
    action = sess.run(agent.policy, {agent.inputs: model_input.reshape(1,-1)})
    time_steps[agent_index] = envs[agent_index].step(action)
    
    rewards.append(time_steps[agent_index].reward)

    if generation_counter == 0 and agent_index == 0 and generation_round == rounds_per_generation:
        image_data = envs[agent_index].physics.render(height = height, width = width, camera_id = 'side')
        img = Image.fromarray(image_data, 'RGB')
        img.save('framesStart/frame-%.10d.png' % time_step_counter)  

    if generation_counter == 250 and agent_index == 0 and generation_round == rounds_per_generation:
        image_data = envs[agent_index].physics.render(height = height, width = width, camera_id = 'side')
        img = Image.fromarray(image_data, 'RGB')
        img.save('framesMiddle/frame-%.10d.png' % time_step_counter)  

    if generation_counter == total_generations - 1 and agent_index == 0 and generation_round == rounds_per_generation:
        image_data = envs[agent_index].physics.render(height = height, width = width, camera_id = 'side')
        img = Image.fromarray(image_data, 'RGB')
        img.save('framesEnd/frame-%.10d.png' % time_step_counter)
        
    time_step_counter += 1

# Below we save the videos from frames recorded during the training loop
image_folders = ['framesStart', 'framesMiddle', 'framesEnd']
video_names = ['videoStart.avi', 'videoMiddle.avi', 'videoEnd.avi']

images = [[img for img in os.listdir(img_f) if img.endswith(".png")] for img_f in image_folders]
frame = cv2.imread(os.path.join(image_folders[0], images[0][0]))
height, width, layers = frame.shape

[imgs.sort(key=natural_keys) for imgs in images]

frames_per_second = 30
videos = [cv2.VideoWriter(vn, -1, frames_per_second, (width,height)) for vn in video_names]

for v in range(len(videos)):
    for image in images[v]:
        videos[v].write(cv2.imread(os.path.join(image_folders[v], image)))

cv2.destroyAllWindows()
[v.release() for v in videos]
