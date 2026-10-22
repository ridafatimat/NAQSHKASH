num_mapping = {
    0: "",
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',
    6: 'f',
    7: 'g',
    8: 'h',
    9: 'i',
    10: 'j',
    20: 'k',
    30: 'l',
    40: 'm',
    50: 'n',
    60: 'o',
    70: 'p',
    80: 'q',
    90: 'r',
    100: 's',
    110: 't',
    120: 'u',
    130: 'v',
    140: 'w',
    150: 'x',
    160: 'y',
    170: 'z',
    180: '{',
    190: '|',
    200: '\'',
    210: '(',
    220: ')',
    230: '*',
    240: '+',
    250: ',',
}



def getTalamCountString(num: int) -> str:
    talam = ""
    
    if (num < 10):
        talam = num_mapping[num]
    elif (num < 260):
        higher = num_mapping[num - (num % 10)]
        lower = num_mapping[num % 10]
        talam = f"{higher}{lower}"
        
    return talam