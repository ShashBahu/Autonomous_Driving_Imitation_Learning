import carla

def main():
    # Connect to CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map = world.get_map()

    # Get all spawn points
    spawn_points = map.get_spawn_points()
    print(f"Found {len(spawn_points)} spawn points:\n")

    for i, sp in enumerate(spawn_points):
        loc = sp.location
        print(f"Spawn {i}: Location(x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f})")

        # Optional: draw in the CARLA world
        world.debug.draw_string(
            loc + carla.Location(z=1.5),
            f"SP {i}",
            draw_shadow=False,
            color=carla.Color(r=0, g=255, b=0),
            life_time=600.0,
            persistent_lines=True
        )

if __name__ == '__main__':
    main()
