import os
import cv2

def generate_video(path):
    '''has an error but can generate video'''
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_name = 'prob.mp4'
    img_list = sorted(os.listdir(path))[1:]
    img_num = sorted([int(img.split('.png')[0].split('_')[1]) for img in img_list])
#     print(img_num)
    
    frame = cv2.imread(path + '/batch_0.png')
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 20, (width, height))

    for i in img_num:
        video.write(cv2.imread(path + '/' + 'batch_' + str(i) + '.png'))  

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    
    path = os.getcwd() + '/testing_only_2023-05-31_043308/board_heatmap'
    generate_video(path)