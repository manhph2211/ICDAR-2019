import cv2
from torchvision import transforms
from model import SSD
import os


def get_test_img_path(test_dir='../data/task12_test',idx=1):
	test_paths=os.listdir(test_dir)
	test_paths_=[os.path.join(test_dir,x) for x in test_paths]
	img_file_path = test_paths_[idx]

	return img_file_path



def show_predict(net,img_file_path):
    img = cv2.imread(img_file_path)
    img_= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_=cv2.resize(img_,(300,300))
    img_=torch.tensor(img_, dtype=torch.float32)
    img_=img_.permute(2, 0, 1)
    transform=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_=transform(img_)
    net.eval()
    input = img_.unsqueeze(0)#(1, 3, 300, 300)

    output = net.apply(input)
    print(output)
    plt.figure(figsize=(10, 10))
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    font = cv2.FONT_HERSHEY_SIMPLEX

    detections = output.data #(1, 2, 200, 5) 5: score, cx, cy, w, h
   
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            cv2.rectangle(img,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          colors[i%3], 2
                          )
            display_text = "%s: %.2f"%(classes[i-1], score)
            cv2.putText(img, display_text, (int(pt[0]), int(pt[1])),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            j += 1
    
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
	net = SSD(phase="inference", cfg=cfg)
	net_weights = torch.load("./task1/ssd300_10.pth", map_location={"cuda:0":"cpu"})
	net.load_state_dict(net_weights)
	show_predict(get_test_img_path())
	