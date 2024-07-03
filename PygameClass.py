import pygame
import numpy as np

class PAINT:
    def __init__(self, x:int, y:int, w:int, h:int, cell_size:int):

        #Matrix for saving the image
        self.array = np.zeros((w, h), dtype = int)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cell_size = cell_size

    def show(self, screen):
        for row in range(self.array.shape[0]):
            for column in range(self.array.shape[1]):
                if self.array[row, column] != 0:
                    light = self.array[row, column]
                    x_pos = int(column * self.cell_size) + self.x
                    y_pos = int(row * self.cell_size) + self.y
                    pixel_rect = pygame.Rect(x_pos, y_pos, self.cell_size, self.cell_size)
                    pygame.draw.rect(screen, (light, light, light), pixel_rect)

    def change_pixel(self, mouse_x, mouse_y, change):
        column = mouse_x // self.cell_size
        row = mouse_y // self.cell_size
        if row - 1 < 0 or row + 1 >= self.w or column - 1 < 0 or column + 1 >= self.h:
            pass
        elif change == 1:
            hardness = 128
            self.array[row, column] += hardness
            self.array[row-1, column-1] += hardness / 4
            self.array[row-1, column] += hardness / 2
            self.array[row-1, column+1] += hardness / 4
            self.array[row, column-1] += hardness / 2
            self.array[row, column+1] += hardness / 2
            self.array[row+1, column-1] += hardness / 4
            self.array[row+1, column] += hardness / 2
            self.array[row+1, column+1] += hardness / 4
            self.array[self.array > 255] = 255
        elif change == 0:
            self.array[row, column] = 0

class TEXT:
    def __init__(self, x:int, y:int, w:int, h:int, font:pygame.font.Font, text:str):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.font = font
        self.text = text
        self.color = pygame.Color('lightskyblue3')
    
    def show(self, screen):

        #Render the text surface.
        self.text_surface = self.font.render(self.text, False, 'White')

        #Get the text rectangle.
        self.rect = self.text_surface.get_rect(center = (self.x,self.y))

        #Text box min size
        self.rect.w = max(self.w, self.text_surface.get_width())

        #Draw the text box.
        pygame.draw.rect(screen, self.color, self.rect, 2)

        #Draw the text.
        screen.blit(self.text_surface, self.rect)

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
    def __init__(self, x:int, y:int, font:pygame.font.Font, image:pygame.image, w_scale:float, h_scale:float, text:str):
        self.x = x
        self.y = y
        self.font = font
        w = image.get_width()
        h = image.get_height()
        self.image = pygame.transform.scale(image, (int(w * w_scale), int(h * h_scale)))
        self.text = text
        self.clicked = False

        self.image_rect = self.image.get_rect(center = (self.x,self.y))
        self.text_surface = self.font.render(self.text, True, 'White')
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