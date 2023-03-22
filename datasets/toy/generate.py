'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:01:54
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:32:33
 # @ Description: This file is distributed under the MIT license.
'''


import numpy as np
import matplotlib.pyplot as plt

import pygame
import os

root = "/Users/melkor/Documents/datasets/"



def random_color():
    color = [0, 10, 0]
    color[np.random.choice([0,2])] = 200
    return color

def random_coord(margin,resolution = (128,128)):
    px = np.random.randint(margin, resolution[0] - margin)
    py = np.random.randint(margin, resolution[1] - margin)
    return px,py

def generate_toy_dataset(num, resolution = (128,128)):
    # Import and initialize the pygame library

    pygame.init()

    # Set up the drawing window
    screen = pygame.display.set_mode(resolution)
    background_image = pygame.image.load("/Users/melkor/Documents/datasets/bg.webp").convert()

    # Run until the user asks to quit
    running = True
    itr = 0
    while running:
        itr += 1

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if itr > num - 1: running = False

        # Fill the background with white
        screen.fill((255, 255, 255))
        screen.blit(background_image, [0, 0])

        for _ in range(np.random.choice([1,2,3])):
            scale = np.random.randint(resolution[0]/12,resolution[0] / 9)
            # choose the color to draw
            color = random_color()
            px,py = random_coord(scale, resolution)

            # control the portion of different kind of objects generated
            category = np.random.choice([0,1,2], p = [0.2, 0.4, 0.4])
            if category == 0:
                # draw tower
                top_color = random_color()
                top_cat = np.random.choice([0,1], p = [0.2,0.8])
                tower_size = scale / 1.2 #np.random.randint(resolution[0]/12,resolution[0]/9)
                if top_cat == 0:
                    # draw the triangle
                    tri_pos = [[px,py],[px+tower_size,py],[px+tower_size/2,py-tower_size/2]]
                    pygame.draw.polygon(screen, top_color, tri_pos)
                if top_cat == 1:
                    pygame.draw.circle(screen, top_color, (px+tower_size/2, py-tower_size/2), tower_size / 2)
                pygame.draw.rect(screen, color, (px,py,tower_size,tower_size*1.5))
                
            if category == 1:
                # draw boat
                top_color = random_color()
                tri_pos = [[px + scale/3, py],[px+scale/2 * 2.0,py - scale],[px+scale/3,py-scale*2]]
                pygame.draw.polygon(screen, top_color, tri_pos)
                pygame.draw.rect(screen, color, (px, py, scale, scale/2))
                
            if category == 2:
                # draw a house
                top_color = random_color()
                scale *= 2
                margin = np.random.randint(0.2 * scale, 0.3 * scale)
                tri_pos = [[px,py],[px+scale,py],[px+scale/2,py-scale/2]]
                pygame.draw.polygon(screen, top_color, tri_pos)
                pygame.draw.rect(screen, color, (px + margin, py, scale - 2 * margin, scale - 2 * margin))

        # Flip the display
        pygame.display.flip()
        pygame.image.save(screen, "{}{}{}.png".format(root,"toy/images/",itr))
    # Done! Time to quit.
    pygame.quit()

if __name__ == "__main__":
    generate_toy_dataset(2000, [256,256])