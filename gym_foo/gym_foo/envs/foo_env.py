import gymnasium as gym
import os
from gymnasium import spaces
import mitsuba as mi
mi.set_variant("scalar_rgb")
import drjit as dr
import PIL.Image as Image
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class FooEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
          low=0, high=255, shape=(100,100, 3), dtype=np.uint8)
        
        self.eye_direction = np.array([0.,0.,0.])
        self.position=np.array([0,1,6.8])
        self.up=np.array([0,1,0])
        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will translate the lookAt target in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1., 0., 0.]),
            1: np.array([0., 1., 0.]),
            2: np.array([-1., 0., 0.]),
            3: np.array([0., -1., 0.]),
            4: np.array([0., 0., 1.]),
            5: np.array([0., 0., -1.]),
        }


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.episode_folder=str(time.time())
        os.makedirs(self.episode_folder)

        self.nb_steps=0#20 steps max per episode

        #reset camera
        self.eye_direction = np.array([0.,0.,0.])
        self.position=np.array([0,1,6.8])
        self.up=np.array([0,1,0])

        # setup the 3D scene
        self.scene,self.params,self.light_pos = self.setup_scene()

        #get the observation (an image)
        img = mi.render(self.scene)
        observation=np.array(mi.Bitmap(img).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True
        ))

        #for updates on performance
        self.first_reward=None

        return observation,{}

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we turn the camera
        direction = self._action_to_direction[action]*.02 #not too big a step
        self.nb_steps+=1
        #rotate camera
        self.eye_direction += direction
        self.params["PerspectiveCamera.to_world"] = mi.Transform4f.look_at(self.position,self.eye_direction,self.up)
        self.scene.parameters_changed()

        #compute reward (it is not sparse at all)
        direc1 = self.eye_direction-self.position
        direc1 = direc1/np.linalg.norm(direc1)
        direc2 = self.light_pos-self.position
        direc2 = direc2/np.linalg.norm(direc2)
        reward = 1-((direc1-direc2)**2).mean()*100#reward : negative unless we're pretty close to the objective (because we start there)

        # An episode is done if the agent has reached the target
        terminated = self.nb_steps>40
        #for displaying the reward gain in each run
        if self.first_reward is None:
            self.first_reward=reward
        if terminated:
            print("reward inc : ",reward - self.first_reward)
        
        #get observation
        img = mi.render(self.scene)
        observation=np.array(mi.Bitmap(img).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True
        ))
        im = Image.fromarray(observation)
        im.save(self.episode_folder + "/" + str(time.time())+".png")
        

        return observation, reward, terminated, False,{}

    def render(self):
        if last_reward is not None:
            print("last reward : ",self.last_reward)
        print("step : ",self.nb_steps)

    def setup_scene(self):
      scene = mi.load_file("scene2.xml")
      light = scene.shapes()[-1]#access the light
      params = mi.traverse(scene)
      #use the lookAt function to change the its position and direction at the center
      position=np.random.rand(3)*np.array([6.,0.5,6.])+np.array([-3.,.1,-3.])
      position /= np.linalg.norm(position)/2.
      params['Light.position'] = position
      params['Light.intensity.value'] *=15.
      params['PerspectiveCamera.to_world']
      scene.parameters_changed()
      
      return scene,params,position
