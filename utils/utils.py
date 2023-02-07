from PIL import Image

def read_txt(path):
    text = []
    with open(path) as f:
        for line in f:
            text.append(line.split('\n')[0])
    N = len(text)
    return text, N

def read_indices(path):
    indices = []
    with open(path) as f:
        for line in f:
            indices.append([int(x) for x in line.split(',')])
    N = len(indices)
    return indices, N

def read_num(path):
    numbers = []
    with open(path) as f:
        for line in f:
            numbers.append(float(line))
    N = len(numbers)
    return numbers, N

def write_txt(lst, opt_path):
    # open file in write mode
    with open(opt_path, 'w') as fp:
        for item in lst:
            # write each item on a new line
            fp.write("%s\n" % item)
        print(f'Text file saved at: {opt_path}')
    return
    
def add_margin(pil_img, pad_x, pad_y, color=128):
    width, height = pil_img.size
    new_width = width + 2*pad_x
    new_height = height + 2*pad_y
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (pad_x, pad_y))
    return result

