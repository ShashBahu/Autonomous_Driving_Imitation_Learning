# The data collector. It connects to the carla server and records driving data
import sys
sys.path.append('/home/krg/CARLA_0.9.13/PythonAPI')
sys.path.append('/home/krg/CARLA_0.9.13/PythonAPI/carla')

from utils.screen import clear, message, warn
from utils.piloterror import PilotError
import numpy as np
import carla, datetime, pygame, os
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
import time

class Collector:
    def __init__(self, world, sim_time):
        self.start_time = datetime.datetime.now()
        self.world = world
        self.vehicle = None
        self.map = self.world.get_map()
        try:
            pygame.init()
        except: pass
        try:
            self.display = pygame.display.set_mode((1900, 1000))
        except:
            warn("Failed to spawn live feed view for data collector. If you're on WSL, this happens as the OS doesn't have a display device yet. Otherwise, check your pygame installation.")
            pass
        self.directory = f'recordings/{datetime.datetime.now().strftime("%Y-%m-%d@%H.%M.%S" if os.name is "nt" else "%Y-%m-%d@%H:%M:%S" )}'
        self.start(sim_time)
    
    def record(self, image):
        control = self.vehicle.get_control()        
        image.save_to_disk(f'{self.directory}/{[int((datetime.datetime.now() - self.start_time).total_seconds()), control.steer, control.throttle, control.brake]}.png')
        
        # we now convert image into a raw image to show in our display
        image.convert(carla.ColorConverter.Raw)
        
        # convert the image into an array using standard procedure
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        # show the frame in display & update it
        try:
            surf = pygame.surfarray.make_surface(np.rot90(array, 1))
            self.display.blit(surf, (0, 0))
            pygame.display.update()
        except:
            warn('Display is not working')

    def start(self, sim_time):

        try:
            # === Manually define loop endpoints: Town01 ===
            ## RIGHT TURN
            #start_location = carla.Location(x=22.18, y=326.97, z=0.30)
            #end_location   = carla.Location(x=1.51, y=295.42, z=0.30)
            
            ## LEFT TURN
            # start_location = carla.Location(x=-1.28, y=309.46, z=0.30)
            # end_location   = carla.Location(x=22.18, y=330.46, z=0.30)

            # === Manually define loop endpoints: Town10HD ===
            # Left Turn
            # start_location = carla.Location(x=-110.96, y=59.69, z=0.60)
            # end_location   = carla.Location(x=106.02, y=50.87, z=0.60)

            # Right Turn
            start_location = carla.Location(x=-13.34, y=-61.05, z=0.60)
            end_location   = carla.Location(x=102.93, y=-9.38, z=0.60)

            # Snap to roads
            start_wp = self.map.get_waypoint(start_location, project_to_road=True)
            end_wp   = self.map.get_waypoint(end_location, project_to_road=True)

            # get a list of spawn points & vehicles then randomly choose from one to use
            message('Spawning vehicle')
            #vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle*')
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprints = blueprint_library.find('vehicle.dodge.charger_police') 
            #spawn_points = self.world.get_map().get_spawn_points()
            #self.vehicle = self.world.spawn_actor(vehicle_blueprints, np.random.choice(spawn_points))
            #for actor in self.world.get_actors().filter('traffic.traffic_light'):
            #    actor.destroy()

            self.vehicle = self.world.spawn_actor(vehicle_blueprints, start_wp.transform)      

            message('OK')
        except Exception as e:
            print(e)
            raise PilotError('Failed to spawn vehicle. Check start() in utils/collect.py for more info')

        try:
            # make a camera and configure it
            message('Spawning camera and attaching to vehicle')
            camera_init_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
            camera_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_blueprint.set_attribute('image_size_x', '720')
            camera_blueprint.set_attribute('image_size_y', '405')
            camera_blueprint.set_attribute('fov', '110') # sets field of view (FOV)
            message('OK')
        except:
            raise PilotError('Failed to attach camera to vehicle. Check start() in utils/collect.py for more info')

        # Set up GRP properly
        grp = GlobalRoutePlanner(self.map, sampling_resolution=2.0)

        # === Generate forward and return route ===
        route_there = grp.trace_route(start_wp.transform.location, end_wp.transform.location)
        #route_back  = grp.trace_route(end_wp.transform.location, start_wp.transform.location)

        # Merge into a loop
        full_loop = route_there # + route_back
        #route = [wp.transform for (wp, _) in full_loop]

        instr = {
            "ignore_traffic_lights":True,
            "ignore_stop_signs":True
        }

        # Setup agent
        agent = BasicAgent(self.vehicle, target_speed=30, opt_dict=instr)
        agent.set_global_plan(full_loop) 

        # attach camera to vehicle and start recording
        self.camera = self.world.spawn_actor(camera_blueprint, camera_init_trans, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.record(image))
        #tm = self.client.get_trafficmanager()
        #tm_port = tm.get_port()

        #self.vehicle.set_autopilot(True, tm_port)
        #tm.ignore_lights_percentage(self.vehicle, 100.0)  # ignore all lights

        # autopilot obviously
        #self.vehicle.set_autopilot(True)

        try:
            elapsed = 0
            # in Carla, we have to call tick() or wait_for_tick() after altering anything in order to reflect change
            while elapsed <= sim_time*60:
                if agent.done():
                    print("Loop complete. Restarting...")
                    # Reinitialize the plan
                    self.vehicle.set_transform(start_wp.transform)
                    self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                    self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                    agent.set_global_plan(full_loop)

                control = agent.run_step()
                self.vehicle.apply_control(control)
                self.world.tick()
                time.sleep(0.1)
                if elapsed != int((datetime.datetime.now() - self.start_time).total_seconds()):
                    elapsed = int((datetime.datetime.now() - self.start_time).total_seconds())
                    clear()
                    message(f'Time elapsed: {int(((datetime.datetime.now() - self.start_time).total_seconds())/60.0)}m {int((datetime.datetime.now() - self.start_time).total_seconds())}s')
            self.stop()
        except KeyboardInterrupt:
            self.stop()
            raise PilotError('You stopped the recording manually. Cleaning up and returning to main menu')
    
    def stop(self):
        message('Quitting recorder')
        try:
            self.camera.stop() # destroy sensor in main smiulation (server)
            self.vehicle.destroy()
        except:
            pass
        message("Vehicle destroyed")
        try:
            pygame.display.quit()
        except: pass
