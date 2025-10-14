import sys
sys.path.append('/home/krg/CARLA_0.9.13/PythonAPI')
sys.path.append('/home/krg/CARLA_0.9.13/PythonAPI/carla')

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner
from carla import WeatherParameters




import time

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map = world.get_map()
    
    weather = WeatherParameters(
    sun_altitude_angle=90.0,  # High noon = minimal shadows
    sun_azimuth_angle=0.0,
    cloudiness=0.0,           # Clear sky (no cloud shadows)
    precipitation=0.0,
    fog_density=0.0,
    wetness=0.0
    )
    world.set_weather(weather)

    # === Manually define loop endpoints ===
    # start_location = carla.Location(x=-110.96, y=59.69, z=0.60)
    # end_location   = carla.Location(x=106.02, y=50.87, z=0.60)

    # RIGHT TURN : Right-lane
    # start_location = carla.Location(x=26.38, y=-57.40, z=0.60)
    # end_location   = carla.Location(x=99.38, y=-6.31, z=0.60) 
    
    # LEFT TURN : Right-Lane
    start_location = carla.Location(x=-114.43, y=56.85, z=0.60)
    end_location   = carla.Location(x=109.52, y=89.84, z=0.60)
    
    # Right Turn
    # start_location = carla.Location(x=-13.34, y=-61.05, z=0.60)
    # end_location   = carla.Location(x=102.93, y=-9.38, z=0.60)

    # Snap to roads
    start_wp = map.get_waypoint(start_location, project_to_road=True)
    end_wp   = map.get_waypoint(end_location, project_to_road=True)


    try:
        # Spawn vehicle at start
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.dodge.charger_police')[0]
        vehicle = world.spawn_actor(vehicle_bp, start_wp.transform)

        # Set up GRP properly
        grp = GlobalRoutePlanner(map, sampling_resolution=2.0)

        # === Generate forward and return route ===
        route_there = grp.trace_route(start_wp.transform.location, end_wp.transform.location)
        #route_back  = grp.trace_route(end_wp.transform.location, start_wp.transform.location)

        # Merge into a loop
        full_loop = route_there # + route_back
        #route = [wp.transform for (wp, _) in full_loop]

        instr = {
            "ignore_traffic_lights":True,
            "ignore_stop_signs":True,
            # 'sampling_resolution':1.0
        }

        # Setup agent
        agent = BasicAgent(vehicle, target_speed=50, opt_dict=instr)
        agent.set_global_plan(full_loop)

        print("Driving full loop (there and back)...")

        while True:
            if agent.done():
                print("Loop complete. Restarting...")
                # Reinitialize the plan
                vehicle.set_transform(start_wp.transform)
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                agent.set_global_plan(full_loop)


            control = agent.run_step()
            vehicle.apply_control(control)
            world.tick()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        vehicle.destroy()
    except Exception as e:
        print(e)
        print("Failed to drive")
        vehicle.destroy()

if __name__ == "__main__":
    main()

