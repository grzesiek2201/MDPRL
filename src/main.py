from world import World


def main():
    world = World(config='map0.txt')
    world.move('up')
    world.show_world('type')

if __name__ == '__main__':
    main()