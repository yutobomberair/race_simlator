import cv2


# obj_size = [573, 861]
obj_size = [574, 862]

img = cv2.imread("./gui_parts/waiting.png")
h, w, _ = img.shape

# diff_w = int((obj_size[1] - w) / 2)
# diff_h = int((obj_size[0] - h) / 2) 

# img_side = img[:, :diff_w]

# img = cv2.hconcat([img_side, img, img_side])
# print(img.shape)

# img_ud = img[:diff_h, :]

# img = cv2.vconcat([img_ud, img, img_ud])
# print(img.shape)

img = cv2.resize(img, (obj_size[1], obj_size[0]))

cv2.imwrite("./resized.png", img)