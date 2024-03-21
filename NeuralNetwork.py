import pygame
import sys
import os
import numpy as np
from PIL import Image
import csv
from Layer import *
np.random.seed(0)

#Find the first missing number in a ordered list.
def find_gap(list):
    length = len(list)
    
    # Check if the list is empty or starts from a number other than 0
    if length == 0 or list[0] != 0:
        return 0
    
    # Iterate through the list to find the first gap
    for i in range(1, length):
        if list[i] != list[i-1] + 1:
            return list[i-1] + 1
    
    # If no gap is found, return the next number in the sequence
    return list[-1] + 1

#Round the array to 3 decimal places.
def round_number(number):
    # Format array elements to display with three decimal places
    if isinstance(number,(int,float)):
        return "{:.3e}".format(number) if abs(number) < 0.001 else "{:.3f}".format(number)
    elif isinstance(number,np.ndarray):
        return ["{:.3e}".format(i) if abs(i) < 0.001 else "{:.3f}".format(i) for i in number]

#########################################################################################################################
#Neural network section


class NEURAL:
    def __init__(self):
        self.n_input = 784
        self.n_output = 10

        self.n_layer = 3
        self.n_neuron = [self.n_input, 32, 32, self.n_output]
        self.layers = []
        
        for i in range(self.n_layer - 1):
            self.layers.append(Hidden_Layer(self.n_neuron[i], self.n_neuron[i+1]))
        self.layers.append(Output_Layer(self.n_neuron[-2],self.n_neuron[-1]))

        self.cost = 0
        self.batch_size = 50
        self.epoch_item = 1
        self.batch_item = 1
        self.rate = 1e-1
        self.correct = 0

        self.mnist_list = []
        self.mnist_index = 0

        with open(MNIST_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i in reader:
                self.mnist_list.append(i)

    def train(self):

        # digit = random.randint(0,9)
        # #Create the 1D array for the one hot encoding desired output.
        # self.desire_output = np.zeros((1, self.n_output))
        # self.desire_output[0, digit] = 1

        # #Get input from pygame drawing pad.
        # image_path = f"./Digits/{digit}/" + random.choice(os.listdir(f"./Digits/{digit}"))
        # self.digit_image = np.asarray(Image.open(image_path))
        # self.layer_input = self.digit_image.reshape(1,-1) / 255

        if self.mnist_index < len(self.mnist_list):
                digit = int(self.mnist_list[self.mnist_index][0])
                self.layer_input = np.asarray(self.mnist_list[self.mnist_index][1:], dtype=int).reshape(1,-1)
                self.desire_output = np.zeros((1, self.n_output))
                self.desire_output[0, digit] = 1

                self.mnist_paint = PAINT(cell_number*cell_size, 0)
                self.mnist_paint.array = self.layer_input.reshape(28,28)
                self.mnist_paint.draw_pixel()

                self.layer_input = (self.layer_input - np.min(self.layer_input)) / (np.max(self.layer_input) - np.min(self.layer_input))
                
                #Forward propagation
                self.layers[0].forward(self.layer_input)
                for i in range(1, self.n_layer):
                    self.layers[i].forward(self.layers[i-1].output)

                #Back propagation
                self.layers[2].backward(self.desire_output)
                for i in range(self.n_layer-1, 0, -1):
                    self.layers[i-1].backward(self.layers[i].dinput)

                if self.batch_item < self.batch_size:
                    self.batch_item += 1

                elif self.batch_item == self.batch_size:

                    #Adjust all parameters in the network.
                    for layer in self.layers:
                        layer.learn(self.rate,self.batch_size)

                    self.batch_item = 1

                    #Print the final prediction of each digit's probability.
                    print(np.max(self.layers[-1].output))
                    print("Output:", np.argmax(self.layers[-1].output))
                    print("Desire:", digit)
                    print("SEM:", round_number(np.std(self.layers[-1].output)))
                
                if np.argmax(self.layers[-1].output) == digit:
                    self.correct += 1
                self.cost = (self.cost + cross_entropy(self.layers[-1].output,self.desire_output)/784) / 2
            
                self.epoch_item += 1
                self.mnist_index += 1
        else:
            self.epoch_item = 1
            self.mnist_index = 0
            self.correct = 0
            np.random.shuffle(self.mnist_list)
            
    #Test a single digit, no training.
    def test(self,array):

        #Calculater the forward output.
        self.layers[0].forward(array.reshape(1,-1))
        for i in range(1, self.n_layer):
            self.layers[i].forward(self.layers[i-1].output)

        #Print the final prediction of each digit's probability.
        print(self.layers[-1].output)
        self.predicted = np.argmax(self.layers[-1].output)
        print("You write a ", self.predicted)

    #Load a trained neural network.
    def load(self,file_path):
        self.layer_1 = 1
        self.layer_2 = 1
        self.layer_output = 1
        #Load w and b with a for loop and consider using .txt/csv for saving data.

    #Save a trained neural network.
    def save(self):
        ...


#########################################################################################################################
#Pygame Section

class PAINT:

    def __init__(self,x,y):

        #Matrix for saving the image
        self.array = np.zeros((cell_number,cell_number), dtype = int)
        self.x = x
        self.y = y

    def draw_pixel(self):
        for row in range(self.array.shape[0]):
            for column in range(self.array.shape[1]):
                if self.array[row,column] != 0:
                    light = self.array[row,column]
                    x_pos = int(column * cell_size) + self.x
                    y_pos = int(row * cell_size) + self.y
                    pixel_rect = pygame.Rect(x_pos,y_pos,cell_size,cell_size)
                    pygame.draw.rect(screen,(light,light,light),pixel_rect)

    def change_pixel(self,mouse_x,mouse_y,change):
        column = mouse_x // cell_size
        row = mouse_y // cell_size
        if row - 1 < 0 or row + 1 >= cell_number or column - 1 < 0 or column + 1 >= cell_number:
            pass
        elif change == 1:
            hardness = 128
            self.array[row,column] += hardness
            self.array[row-1,column-1] += hardness/4
            self.array[row-1,column] += hardness/2
            self.array[row-1,column+1] += hardness/4
            self.array[row,column-1] += hardness/2
            self.array[row,column+1] += hardness/2
            self.array[row+1,column-1] += hardness/4
            self.array[row+1,column] += hardness/2
            self.array[row+1,column+1] += hardness/4
            self.array[self.array > 255] = 255
        elif change == 0:
            self.array[row,column] = 0

class TEXT:

    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.text = text
        self.color = pygame.Color('lightskyblue3')
    
    def show(self,screen):

        #Render the text surface.
        self.text_surface = base_font.render(self.text, False, 'White')

        #Get the text rectangle.
        self.rect = self.text_surface.get_rect(center = (self.x,self.y))

        #Text box min size
        self.rect.w = max(self.w, self.text_surface.get_width())

        #Draw the text box.
        pygame.draw.rect(screen,self.color,self.rect,2)

        #Draw the text.
        screen.blit(self.text_surface,self.rect)

    def input(self,event):
        action = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                action = True
            else:
                self.text += event.unicode

        return action

class BUTTON:

    def __init__(self, x:int, y:int, image:pygame.image, w_scale:float, h_scale:float, text:str):
        self.x = x
        self.y = y
        w = image.get_width()
        h = image.get_height()
        self.image = pygame.transform.scale(image, (int(w * w_scale), int(h * h_scale)))
        self.text = text
        self.clicked = False

        self.image_rect = self.image.get_rect(center = (self.x,self.y))
        self.text_surface = base_font.render(self.text, True, 'White')
        self.text_rect = self.text_surface.get_rect(center = (self.x, self.y))

    def show(self, surface:pygame.Surface) -> bool:
        action = False

        #Get mouse position.
        mouse_pos = pygame.mouse.get_pos()

        #Check mouse collision and clicked
        if self.image_rect.collidepoint(mouse_pos):
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                self.clicked = True
                action = True
        
        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        surface.blit(self.image, self.image_rect)
        surface.blit(self.text_surface, self.text_rect)

        return action


#########################################################################################################################
#Pygame game state section

class STATE:

    def __init__(self):
        self.state = "start"
        self.test = False

    #Update the scene.
    def update(self):
        if self.state == "start":
            pygame.display.set_caption("Start")
            self.start()

        if self.state == "input_text":
            if not self.test:
                pygame.display.set_caption("Input text")
            else:
                pygame.display.set_caption("Test input text")
            
            self.input_text()

        if self.state == "drawing_pad":
            pygame.display.set_caption("Drawing pad")
            self.drawing_pad()

        if self.state == "train":
            pygame.display.set_caption("Neural network is training...")
            self.train()

        if self.state == "test_drawing_pad":
            pygame.display.set_caption("Test drawing pad")
            self.test_drawing_pad()
        
        if self.state == "test_output":
            pygame.display.set_caption("Test output")
            self.test_output()
    
    #The start scene.
    def start(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        screen.fill('White')

        background = pygame.Surface((cell_number * cell_size, cell_number * cell_size))
        background.fill("Black")
        screen.blit(background,(0,0))

        #Change to the input scene
        start_button = BUTTON(cell_number * cell_size / 2,(cell_number - 5) * cell_size / 2,border_1,2,0.5,"Start")
        if start_button.show(screen):
            self.state = "input_text"
        
        #Change to the train state
        train_button = BUTTON(cell_number * cell_size / 2,cell_number * cell_size / 2,border_1,2,0.5,"Train")
        if train_button.show(screen):
            self.state = "train"
        
        #Load pre-trained neural network
        load_button = BUTTON(cell_number * cell_size / 2,(cell_number + 5) * cell_size / 2,border_1,2,0.5,"Load")
        if load_button.show(screen):
            #Add loading network code.
            ...
        
        #Quit the game
        quit_button = BUTTON(cell_number * cell_size / 2,(cell_number + 10) * cell_size / 2,border_1,2,0.5,"Quit")
        if quit_button.show(screen):
            pygame.quit()
            sys.exit()
    
    #The text input scene.
    def input_text(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            #Detect user input to the text.
            #Change to the drawing pad scene when enter is inputted.
            if text.input(event):
                self.state = "drawing_pad"
        
        screen.fill('White')

        background = pygame.Surface((cell_number * cell_size, cell_number * cell_size))
        background.fill("Black")
        screen.blit(background,(0,0))

        text.show(screen)
        prompt.show(screen)

    #The drawing scene.
    def drawing_pad(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:

                #Press key Enter to draw the next digit.
                if event.key == pygame.K_RETURN:

                    #Save the drawing to corresponding digit folder.
                    im = Image.fromarray(paint.array.astype('uint8'), mode = 'L')
                    digit_list = os.listdir(f"./Digits/{text.text}")
                    for i,file in enumerate(digit_list):
                        digit_list[i] = int((file.split('.')[0]).split('_')[1])
                    digit_list.sort()
                    gap = find_gap(digit_list)
                    im.save(f"./Digits/{text.text}/{text.text}_{gap}.jpg")

                    self.state = "input_text"
                
                if event.key == pygame.K_t:
                    self.state = "train"
                
                paint.array = np.zeros((cell_number,cell_number), dtype=int)
                text.text = ''
                    
        #Detect mouse left click.
        if pygame.mouse.get_pressed()[0] == True:
            mouse_x,mouse_y = pygame.mouse.get_pos()
            mouse_x = max(0, min(mouse_x, cell_number * cell_size))
            mouse_y = max(0, min(mouse_y, cell_number * cell_size))
            paint.change_pixel(mouse_x,mouse_y,1)
        
        #Detect mouse right click.
        elif pygame.mouse.get_pressed()[2] == True:
            mouse_x,mouse_y = pygame.mouse.get_pos()
            paint.change_pixel(mouse_x,mouse_y,0)
        
        screen.fill('White')
        
        background = pygame.Surface((cell_number * cell_size, cell_number * cell_size))
        background.fill("Black")
        screen.blit(background,(0,0))
        
        paint.draw_pixel()

    #The back propagation training scene.
    def train(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                #Press key Enter to stop back propagation.
                if event.key == pygame.K_RETURN:
                    self.state = "start"
        
        screen.fill('Black')
        neural.train()
        
        epoch_item = TEXT(cell_number*cell_size/2, (cell_number - 10)*cell_size/2, cell_size,cell_size, "Sample No." + str(neural.epoch_item))
        epoch_item.show(screen)

        cost = TEXT(cell_number*cell_size/2, (cell_number - 5)*cell_size/2, cell_size,cell_size, "Cost: ")
        cost.show(screen)

        numCost = TEXT(cell_number*cell_size/2, (cell_number)*cell_size/2, cell_size,cell_size, str(round_number(neural.cost)))
        numCost.show(screen)

        correct = TEXT(cell_number*cell_size/2, (cell_number + 5)*cell_size/2, cell_size,cell_size, "Correct: " + str(neural.correct))
        correct.show(screen)
        
        correct_rate = TEXT(cell_number*cell_size/2, (cell_number + 10)*cell_size/2, cell_size,cell_size, str(round_number(neural.correct/neural.epoch_item)))
        correct_rate.show(screen)

        #Change to the test drawing pad scene when Test button is clicked.
        test_button = BUTTON(cell_number*cell_size/2, (cell_number + 15)*cell_size/2, border_1, 2, 0.5, "Test")
        if test_button.show(screen):
            self.test = True
            self.state = "test_drawing_pad"
        
        #Save the trained neural network when Save button is clicked.
        save_button = BUTTON(cell_number*cell_size/2, (cell_number + 20)*cell_size/2, border_1, 2, 0.5, "Save")
        if save_button.show(screen):
            #Add saving network code.
            ...

    #The test drawing pad scene.
    def test_drawing_pad(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:

                #Press key Enter to test for this single digit.
                if event.key == pygame.K_RETURN:
                    #Calculate the predicted digit.
                    neural.test(paint.array)

                    #Clear the drawing pad.
                    paint.array = np.zeros((cell_number,cell_number), dtype=int)

                    #Test the neural network.
                    self.state = "test_output"
                    
        #Detect mouse left click.
        if pygame.mouse.get_pressed()[0] == True:
            mouse_x,mouse_y = pygame.mouse.get_pos()
            mouse_x = max(0, min(mouse_x, cell_number * cell_size))
            mouse_y = max(0, min(mouse_y, cell_number * cell_size))
            paint.change_pixel(mouse_x,mouse_y,1)
        
        #Detect mouse right click.
        elif pygame.mouse.get_pressed()[2] == True:
            mouse_x,mouse_y = pygame.mouse.get_pos()
            paint.change_pixel(mouse_x,mouse_y,0)
        
        screen.fill('White')
        
        background = pygame.Surface((cell_number * cell_size, cell_number * cell_size))
        background.fill("Black")
        screen.blit(background,(0,0))
        
        paint.draw_pixel()

    #The test output scene.
    def test_output(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:

                #Press key Enter to test for this single digit.
                if event.key == pygame.K_RETURN:

                    #Test the neural network.
                    self.state = "test_drawing_pad"
    
        screen.fill('White')
        
        background = pygame.Surface((cell_number * cell_size, cell_number * cell_size))
        background.fill("Black")
        screen.blit(background,(0,0))

        predicted = TEXT(cell_number * cell_size / 2,cell_number * cell_size / 2,cell_size,cell_size, "You write a " + str(neural.predicted))
        predicted.show(screen)



#########################################################################################################################
#Pygame initialize section

pygame.init()

base_font = pygame.font.Font("Font/Monocraft.ttf", 32)
cell_size = 15
cell_number = 28
screen = pygame.display.set_mode((1200,600))
clock = pygame.time.Clock()
pygame.display.set_caption("Draw")
border_1 = pygame.image.load("Image/border_4.png").convert_alpha()

MNIST_path = './MNIST/mnist_test.csv'

state = STATE()
paint = PAINT(0,0)
neural = NEURAL()

text = TEXT(cell_number * cell_size / 2,(cell_number + 2) * cell_size / 2,cell_size,cell_size, '')
prompt = TEXT(cell_number * cell_size / 2,cell_number * cell_size / 2 - cell_size,cell_size,cell_size, "Input a digit:")


#########################################################################################################################
#Pygame running section

while True:
    state.update()

    pygame.display.update()
    clock.tick(60)