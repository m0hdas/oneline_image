from PIL import Image, ImageFilter
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy_numpy import solve_tsp  
import turtle

try:
    image = Image.open('neil.jpg') #put file location here
except FileNotFoundError:
    print('File not found.')
    quit()

width, height = image.size
    
bw_image = image.convert('1', dither=Image.NONE)
bw_image_array = np.array(bw_image, dtype=np.int)  
black_indices = np.argwhere(bw_image_array == 0)


chosen_black_indices = black_indices[  
                           np.random.choice(black_indices.shape[0],  
                                            replace=False,  
                                            size= min(10000, len(black_indices) )
                                            )]
            #increasing size improves res but slows down performance

distances = pdist(chosen_black_indices)  
distance_matrix = squareform(distances) 

optimized_path = solve_tsp(distance_matrix)  
optimized_path_points = [chosen_black_indices[x] for x in optimized_path]

turtle.screensize(640, 360) #change as you like
turtle.penup()
turtle.goto(optimized_path_points[0][1] - width//2, height//2 - optimized_path_points[0][0])
turtle.pendown()
for i in optimized_path_points:
    turtle.goto(i[1] - width//2, height//2 - i[0])

turtle.hideturtle()
