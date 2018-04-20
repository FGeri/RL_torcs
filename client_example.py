#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from critic import Critic
from actor import Actor
from experience_memory import ExperienceMemory
from copy import deepcopy
from carla.client import make_carla_client,VehicleControl
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

# =============================================================================
#  Subtaskok makrói
# =============================================================================
SUBTASK_DRIVE_IN_LANE = 1
SUBTASK_TURN_AROUND = 2
SUBTASK_TURN_LEFT = 3
SUBTASK_TURN_RIGHT = 4

# =============================================================================
# Hyperparams
# =============================================================================
CRITIC_LR = 0.001
ACTOR_LR = 0.0001

MINI_BATCH_SIZE = 128

TARGET_UPDATE_BASE_FREQ = 100
TARGET_UPDATE_MULTIPLIER = 1.05
GAMMA = 0.95


def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 60000
    frames_per_episode = 400



    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port,30) as client:
        print('CarlaClient connected')


# =============================================================================
#       Global initialisations
# =============================================================================
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

        state_size={'state_2D':(64,64,9,),
                    'state_1D':(17,)}
        action_size=(5,)


        critic = Critic(sess,state_size,action_size,CRITIC_LR)
        critic.target_train()
        actor = Actor(sess,state_size,action_size,ACTOR_LR)
        actor.target_train()
        memory = ExperienceMemory(100000,False)


        target_update_counter = 0
        target_update_freq = TARGET_UPDATE_BASE_FREQ

        explore_rate = 0.2

        success_counter = 0

        total_t = 0
        t=0
        #NOTE Ez csak egy próba, eztmég át kell alakítani
        target = {'pos':np.array([-3.7,236.4,0.9]),
                  'ori':np.array([0.00,-1.00,0.00])}

        if args.settings_filepath is None:
            # Create a CarlaSettings object. This object is a wrapper around
            # the CarlaSettings.ini file. Here we set the configuration we
            # want for the new episode.
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=0,
                NumberOfPedestrians=0,
                WeatherId=random.choice([1]),
                QualityLevel=args.quality_level)
#            settings.randomize_seeds()
#
#            settings.randomize_seeds()
            # The default camera captures RGB images of the scene.
            camera0 = Camera('CameraRGB')
            # Set image resolution in pixels.
            camera0.set_image_size(64, 64)
            # Set its position relative to the car in centimeters.
            camera0.set_position(0.30, 0, 1.30)
            settings.add_sensor(camera0)
        else:

            # Alternatively, we can load these settings from a file.
            with open(args.settings_filepath, 'r') as fp:
                settings = fp.read()
        scene = client.load_settings(settings)

# =============================================================================
#       EPISODES LOOP
# =============================================================================
        for episode in range(0, number_of_episodes):
            # Start a new episode.
            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))
            player_start = 0
            total_reward = 0.
            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode...')
            client.start_episode(player_start)



            #TODO Ezen belül kéne implementálni a tanuló algoritmusunkat

# =============================================================================
#           Episodic intitialisations
# =============================================================================
            collisions ={'car':0,'ped':0,'other':0}
            reverse = -1.0
            measurements, sensor_data = client.read_data()
            state = get_state_from_data(measurements, sensor_data,reverse)
            goal = get_goal_from_data(target)
            t=0
            stand_still_counter=0
# =============================================================================
#           STEPS LOOP
# =============================================================================
            for frame in range(0, frames_per_episode):
                t=t+1
                total_t+=1
                target_update_counter+=1
                explore_dev = 0.6/(1+total_t/30000)
                explore_rate = 0.3/(1+total_t/30000)
            # Print some of the measurements.
            #   print_measurements(measurements)

                # Save the images to disk if requested.
                if args.save_images_to_disk and False:
                    for name, measurement in sensor_data.items():
                        filename = args.out_filename_format.format(episode, name, frame)
                        measurement.save_to_disk(filename)

                if state['state_1D'][9]<5 and t>50:
                    stand_still_counter+=1
                else:
                    stand_still_counter=0
                #Calculate the action
                a_pred = actor.model.predict([np.expand_dims(state['state_2D'],0),
                                              np.expand_dims(np.concatenate((state['state_1D'],goal)),0)])[0]
                #Add exploration noise to action
                a = add_noise(a_pred,explore_dev,explore_rate)
                control = get_control_from_a(a)
                #Sendcontrol to the server
                client.send_control(control)

#
# =============================================================================
#               TRAINING THE NETWORKS
# =============================================================================
                if memory.num_items > 6000:
                    batch,indeces = memory.sample_experience(MINI_BATCH_SIZE)
                    raw_states = [[e[0]['state_2D'],e[0]['state_1D']] for e in batch]
                    goals = np.asarray([e[5] for e in batch])
                    states = {'state_2D':np.atleast_2d(np.asarray([e[0] for e in raw_states[:]])),
                              'state_1D':np.atleast_2d(np.asarray([np.concatenate([e[1],goals[i]],axis=-1) for i,e in enumerate(raw_states[:])]))}

                    actions = np.asarray([e[1] for e in batch])
                    rewards = np.asarray([np.sum(e[2]) for e in batch]).reshape(-1,1)


                    raw_new_states = [[e[3]['state_2D'],e[3]['state_1D']] for e in batch]
                    new_states = {'state_2D':np.atleast_2d(np.asarray([e[0] for e in raw_new_states[:]])),
                                  'state_1D':np.atleast_2d(np.asarray([np.concatenate([e[1],goals[i]],axis=-1) for i,e in enumerate(raw_new_states[:])]))}

                    overs = np.asarray([e[4] for e in batch]).reshape(-1,1)



                    best_a_preds = actor.target_model.predict([new_states['state_2D'],new_states['state_1D']])
                    max_qs = critic.target_model.predict([new_states['state_2D'],
                                                          new_states['state_1D'],
                                                          best_a_preds])

                    ys = rewards + (1-overs)*GAMMA*max_qs
                    #Train Critic network
                    critic.model.train_on_batch([states['state_2D'],
                                                 states['state_1D'],
                                                 actions],ys)
                    #Train Actor network
                    a_for_grads = actor.model.predict([states['state_2D'],
                                                       states['state_1D']])
                    a_grads = critic.gradients(states,a_for_grads)
                    actor.train(states,a_grads)

                    #Train target networks
                    if target_update_counter >= int(target_update_freq):
                        target_update_counter = 0
                        target_update_freq = target_update_freq * TARGET_UPDATE_MULTIPLIER
                        critic.target_train()
                        actor.target_train()
# =============================================================================
#               GET AND STORE OBSERVATIONS
# =============================================================================
                #Get next measurements
                measurements, sensor_data = client.read_data()
                new_state = get_state_from_data(measurements, sensor_data, reverse, state)

                #TODO Calculate reward
                r_goal, success = calculate_goal_reward(np.atleast_2d(new_state['state_1D']),goal)
                r_general, collisions = calculate_general_reward(measurements,collisions)
                over = stand_still_counter>30 or success
                success_counter+=int(bool(success)*1)
                total_reward+=r_goal
                total_reward+=r_general
                #Store observation
                if t>10:
                    experience =  pd.DataFrame([[state,a,np.array([r_goal,r_general]),new_state,bool(over),goal,episode,0]],columns=['s', 'a', 'r', "s'", 'over','g','e','p'],copy = True)
                    memory.add_experience(experience)

                #Set the state to the next state
                state = new_state
                if over:
                    break
            sub_goal = deepcopy(state['state_1D'][0:6])
            print(str(episode)+". Episode###################")
            print("Total reward: "+str(total_reward))
            print("Success counter: "+str(success_counter))
            if (episode%10==0):
                print("############## DEBUG LOG ################")
                print("Memory state: "+str(memory.num_items))
                print("Target update counter: "+str(target_update_counter))
                print("Exploration rate: "+str(explore_rate))
                print("Exploration dev: "+str(explore_dev))
                print("Total timesteps: "+str(total_t))
                print("Average episode length: "+str(total_t/(episode+1)))
                print("#########################################")
# =============================================================================
#           REPLAY FOR SUBGOALS
# =============================================================================
            batch = memory.get_last_episode(t)
            raw_new_states = [[e[3]['state_2D'],e[3]['state_1D']] for e in batch]
            new_states = {'state_2D':np.atleast_2d(np.asarray([e[0] for e in raw_new_states[:]])),
                          'state_1D':np.atleast_2d(np.asarray([e[1] for e in raw_new_states[:]]))}
            rewards = np.asarray([e[2] for e in batch]).reshape(-1,2)
            r_subgoal = calculate_goal_reward(new_states['state_1D'],sub_goal)[0]
            rewards[:,0]=r_subgoal
            subgoal_batch = [[v[0],v[1],list(rewards)[i],v[3],v[4],sub_goal,v[6],v[7]] for i,v in enumerate(batch)]
            experiences =  pd.DataFrame(subgoal_batch,columns=['s', 'a', 'r', "s'", 'over','g','e','p'],copy = True)
            memory.add_experience(experiences)

def add_noise(a_pred,dev,explore_rate):
        noise = np.random.normal(0,dev,a_pred.shape)
        a = a_pred+noise
        a[0]=np.clip(a[0],-1.0,1.0)
        a[1]=np.clip(np.clip(a[1],0.0,1.0)+0.3,0.0,1.0)
        a[2]=np.clip(a[2]-0.8,0.0,1.0)
#        a[3]=1 if a[3]>=0.95 else 0
#        a[4]=1 if a[4]>=1 else 0
        a[3]=0
        a[4]=0

        if random.uniform(0,1) < explore_rate:
            a[0]=random.uniform(-1.0,1.0)
            a[1]=random.uniform(0.5,1)
        return a

def get_control_from_a(a):
    control = VehicleControl()
    control.steer=a[0]
    control.throttle=a[1]
    control.brake=a[2]
    control.hand_brake=bool(a[3])
    control.reverse=bool(a[4])
    return control

def get_goal_from_data(target):

        #Goal
    goal = np.hstack([target['pos'][0],target['pos'][1],target['pos'][2],
                      target['ori'][0],target['ori'][1],target['ori'][2]])
    return goal;

def get_state_from_data(measurements,sensor_data,reverse,prev_state=None):
    player_measurements = measurements.player_measurements

    pos_x=player_measurements.transform.location.x
    pos_y=player_measurements.transform.location.y
    pos_z=player_measurements.transform.location.z

    ori_x = player_measurements.transform.orientation.x
    ori_y = player_measurements.transform.orientation.y
    ori_z = player_measurements.transform.orientation.z

    acc_x = player_measurements.acceleration.x
    acc_y = player_measurements.acceleration.x
    acc_z = player_measurements.acceleration.x

    speed=player_measurements.forward_speed * 3.6



#    TODO Befejezni
#    NOTE Nem kéne bele a target pozi is? Esetleg az idő?
    state_1D = np.hstack([pos_x,pos_y,pos_z,
                          ori_x,ori_y,ori_z,
                          acc_x,acc_y,acc_z,
                          speed,reverse])
    if prev_state is None:
        state_2D = np.concatenate([sensor_data['CameraRGB'].data/255*2-1,
                                   sensor_data['CameraRGB'].data/255*2-1,
                                   sensor_data['CameraRGB'].data/255*2-1],-1)
    else:
        state_2D = np.concatenate([sensor_data['CameraRGB'].data/255*2-1,np.array(prev_state['state_2D'][:,:,0:-3])],-1)

    state={'state_1D':state_1D,
           'state_2D':state_2D}

    return state


#Caculate the reward for the subtask drive in lane
def calculate_goal_reward(state_1D,goal):
    reward = 0

    success=np.array([
        np.linalg.norm(state_1D[:,0:3]-goal[0:3],axis=1) < 1,
        np.linalg.norm(state_1D[:,3:6]-goal[3:6],axis=1) < 0.1]);

    reward = np.all(success,axis=0)*10

    return reward,np.all(reward>0)
#TODO TURN AORUND
#TODO OVERCOME
#TODO AVOID

#Calculate the general reward (collision,leaving the lane, etc.)
def calculate_general_reward(measurements,collisions):
    reward = 0
    player_measurements = measurements.player_measurements

#   Collisions
    ped_coeff = 0.3
    car_coeff = 0.2
    other_coeff = 0.1

    new_collisions={'car':player_measurements.collision_vehicles,
                    'ped':player_measurements.collision_pedestrians,
                    'other':player_measurements.collision_other
                    }

    col_cars = new_collisions['car'] - collisions['car']
    col_ped = new_collisions['ped'] - collisions['ped']
    col_other = new_collisions['other'] - collisions['other']
    col_punishment = np.clip(col_ped * ped_coeff + col_cars*car_coeff + col_other*other_coeff,
                             0.,
                             10.)
    reward -= col_punishment

#NOTE Itt is differenciálisan kéne büntetni
#   Leaving the lane
    offroad_coeff = 1
    other_lane_coeff = 1
    other_lane = player_measurements.intersection_otherlane,
    offroad = player_measurements.intersection_offroad,
    offlane_punishment = offroad * offroad_coeff + other_lane * other_lane_coeff
    offlane_punishment = np.asarray(offlane_punishment).sum()
    offlane_punishment = np.clip(offlane_punishment,0.,2.)

    reward-=offlane_punishment
#TODO Standstill punishment
    standstill_coeff = 1.0
    standstill_limit = 5.00

    standstill_punishment = (player_measurements.forward_speed*3.6<standstill_limit) * standstill_coeff
    reward-=standstill_punishment
#TODO Exceeding speed limit
#TODO Crossing a red semaphore
    return reward, new_collisions

def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.2f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x / 100, # cm -> m
        pos_y=player_measurements.transform.location.y / 100,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='152.66.240.18',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=4000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
        print("Ready")
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
