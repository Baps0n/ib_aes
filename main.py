import numpy as np

S_BOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]
INV_S_BOX = [
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]


def rcon_gen():
    rcon = np.empty((4, 14), dtype="int")
    rcon[0] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d]
    for i in range(1, 4):
        for j in range(14):
            rcon[i][j] = 0
    return rcon


def fill_state(inp):
    state = np.empty((4, 4), dtype="int")
    for i in range(4):
        for j in range(4):
            state[i][j] = inp[i + 4 * j]
    return state


def shift_rows(state):
    state[1] = np.roll(state[1], -1)
    state[2] = np.roll(state[2], -2)
    state[3] = np.roll(state[3], -3)
    return state


def inv_shift_rows(state):
    state[1] = np.roll(state[1], 1)
    state[2] = np.roll(state[2], 2)
    state[3] = np.roll(state[3], 3)
    return state


def sub_bytes(state):
    for i in range(4):
        for j in range(4):
            # print(S_BOX[state[i][j]], int(S_BOX[state[i][j]], 0))
            state[i][j] = S_BOX[state[i][j]]
    return state


def inv_sub_bytes(state):
    for i in range(4):
        for j in range(4):
            state[i][j] = INV_S_BOX[state[i][j]]
    return state


def g_mul_256(num, mul):
    if mul == '01':
        return num
    if mul == '02':
        if num < 128:
            return (num << 1) & 0xff
        else:
            return ((num << 1) & 0xff) ^ 0x1b
    if mul == '03':
        return g_mul_256(num, '02') ^ num
    if mul == '09':
        return g_mul_256(g_mul_256(g_mul_256(num, '02'), '02'), '02') ^ num
    if mul == '0b':
        return g_mul_256(g_mul_256(g_mul_256(num, '02'), '02'), '02') ^ g_mul_256(num, '02') ^ num
    if mul == '0d':
        return g_mul_256(g_mul_256(g_mul_256(num, '02'), '02'), '02') ^ g_mul_256(g_mul_256(num, '02'), '02') ^ num
    if mul == '0e':
        return g_mul_256(g_mul_256(g_mul_256(num, '02'), '02'), '02') ^ g_mul_256(g_mul_256(num, '02'), '02') ^ g_mul_256(num, '02')


def mix_columns(state):
    mix_state = np.empty((4, 4), dtype="int")
    for i in range(4):
        mix_state[0][i] = g_mul_256(state[0][i], '02') ^ g_mul_256(state[1][i], '03') ^ g_mul_256(state[2][i], '01') ^ g_mul_256(state[3][i], '01')
        mix_state[1][i] = g_mul_256(state[0][i], '01') ^ g_mul_256(state[1][i], '02') ^ g_mul_256(state[2][i], '03') ^ g_mul_256(state[3][i], '01')
        mix_state[2][i] = g_mul_256(state[0][i], '01') ^ g_mul_256(state[1][i], '01') ^ g_mul_256(state[2][i], '02') ^ g_mul_256(state[3][i], '03')
        mix_state[3][i] = g_mul_256(state[0][i], '03') ^ g_mul_256(state[1][i], '01') ^ g_mul_256(state[2][i], '01') ^ g_mul_256(state[3][i], '02')
    return mix_state


def inv_mix_columns(state):
    inv_mix_state = np.empty((4, 4), dtype="int")
    for i in range(4):
        inv_mix_state[0][i] = g_mul_256(state[0][i], '0e') ^ g_mul_256(state[1][i], '0b') ^ g_mul_256(state[2][i], '0d') ^ g_mul_256(state[3][i], '09')
        inv_mix_state[1][i] = g_mul_256(state[0][i], '09') ^ g_mul_256(state[1][i], '0e') ^ g_mul_256(state[2][i], '0b') ^ g_mul_256(state[3][i], '0d')
        inv_mix_state[2][i] = g_mul_256(state[0][i], '0d') ^ g_mul_256(state[1][i], '09') ^ g_mul_256(state[2][i], '0e') ^ g_mul_256(state[3][i], '0b')
        inv_mix_state[3][i] = g_mul_256(state[0][i], '0b') ^ g_mul_256(state[1][i], '0d') ^ g_mul_256(state[2][i], '09') ^ g_mul_256(state[3][i], '0e')
    return inv_mix_state


def get_nr(nk):
    if nk == 4:
        return 10
    if nk == 6:
        return 12
    if nk == 8:
        return 14


def key_expansion(key):
    nk = len(key) // 4
    rcon = rcon_gen()

    w = np.empty((4, get_nr(nk) * 4 + 4), dtype="int")
    for i in range(4):
        for j in range(nk):
            w[i][j] = key[i + 4 * j]

    for i in range(nk, get_nr(nk) * 4 + 4):
        if i % nk == 0:
            w_prev = [w[1][i - 1], w[2][i - 1], w[3][i - 1], w[0][i - 1]]
            for j in range(4):
                w_prev[j] = S_BOX[w_prev[j]]
                w[j][i] = w[j][i - nk] ^ w_prev[j] ^ rcon[j][i // nk - 1]
        elif (i - 4) % nk == 0 and nk == 8:
            for j in range(4):
                w[j][i] = w[j][i - nk] ^ S_BOX[w[j][i - 1]]
        else:
            for j in range(4):
                w[j][i] = w[j][i - nk] ^ w[j][i - 1]
    return w


def add_round_key(state, key):
    for i in range(4):
        for j in range(4):
            state[i][j] ^= key[i][j]
    return state


def get_key_block(keys, num):
    key = np.empty((4, 4), dtype="int")
    for j in range(4):
        key[j] = keys[j][4 * num:4 * num + 4]
    return key


def aes_encrypt(inp, key):
    nk = len(key) // 4
    keys = key_expansion(key)
    state = fill_state(inp)
    nr = get_nr(nk)

    state = add_round_key(state, get_key_block(keys, 0))
    for i in range(1, nr):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, get_key_block(keys, i))

    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, get_key_block(keys, nr))

    res = list(range(16))
    for i in range(4):
        for j in range(4):
            res[i + 4 * j] = state[i][j]
    return res


def aes_decrypt(inp, key):
    nk = len(key) // 4
    keys = key_expansion(key)
    state = fill_state(inp)
    nr = get_nr(nk)

    state = add_round_key(state, get_key_block(keys, nr))
    for i in reversed(range(1, nr)):
        state = inv_sub_bytes(state)
        state = inv_shift_rows(state)
        state = add_round_key(state, get_key_block(keys, i))
        state = inv_mix_columns(state)

    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    state = add_round_key(state, get_key_block(keys, 0))

    res = list(range(16))
    for i in range(4):
        for j in range(4):
            res[i + 4 * j] = state[i][j]
    return res


def data_split(inp, mode='str'):
    inp_s = [inp[i:i+16] for i in range(0, len(inp), 16)]
    if len(inp_s[-1]) < 16:
        inp_s[-1] += str('a' * (16 - len(inp_s[-1:])))
    for i in range(len(inp_s)):
        inp_s[i] = list(inp_s[i])
        for j in range(len(inp_s[i])):
            if mode == 'str':
                inp_s[i][j] = ord(inp_s[i][j])
            if mode == 'byte':
                inp_s[i][j] = int(inp_s[i][j], 0)
    return inp_s


def data_readable(inp):
    for i in range(len(inp)):
        for j in range(16):
            inp[i][j] = chr(inp[i][j])
        inp[i] = ''.join(inp[i])
    return ''.join(inp)


def data_bytes(inp):
    res = []
    for i in range(len(inp)):
        for j in range(16):
            res.append('0x' + format(ord(inp[i][j]), "02x"))
    return ''.join(res)


def xor_blocks(a, b):
    res = []
    for i in range(16):
        res.append(a[i] ^ b[i])
    return res


def ecb_encrypt(data, key):
    res = []
    for i in data:
        res.append(aes_encrypt(i, key))
    return res


def ecb_decrypt(enc_data, key):
    res = []
    for i in enc_data:
        res.append(aes_decrypt(i, key))
    return res


def cbc_encrypt(data, key, iv):
    res = [iv]
    for i in data:
        res.append(aes_encrypt(xor_blocks(i, res[-1]), key))
    return res[1:]


def cbc_decrypt(enc_data, key, iv):
    enc_data.insert(0, iv)
    res = []
    for i in range(1, len(enc_data)):
        res.append(xor_blocks(enc_data[i-1], aes_decrypt(enc_data[i], key)))
    return res


def cfb_encrypt(data, key, iv):
    res = [xor_blocks(aes_encrypt(iv, key), data[0])]
    for i in range(1, len(data)):
        res.append(xor_blocks(data[i], aes_encrypt(res[i-1], key)))
    return res


def cfb_decrypt(enc_data, key, iv):
    res = [xor_blocks(aes_encrypt(iv, key), enc_data[0])]
    for i in range(1, len(enc_data)):
        res.append(xor_blocks(aes_encrypt(enc_data[i-1], key), enc_data[i]))
    return res


def ofb_process(data, key, iv):
    iv_o = [aes_encrypt(iv, key)]
    res = [xor_blocks(iv_o[0], data[0])]
    for i in range(1, len(data)):
        iv_o.append(aes_encrypt(iv_o[i-1], key))
        res.append(xor_blocks(data[i], iv_o[i]))
    return res


def ctr_process(data, key, iv):
    count = [0]*16
    ctr = [xor_blocks(iv, count)]
    res = [xor_blocks(ctr[0], data[0])]
    for i in range(1, len(data)):
        count[-1] += 1
        ctr.append(aes_encrypt(xor_blocks(iv, count), key))
        res.append(xor_blocks(data[i], ctr[i]))
    return res


def main():
    print("Выберите режим работы, указав его номер:\n"
          "1. ECB - Electronic Codebook\n"
          "2. CBC - Cipher Block Chaining\n"
          "3. CFB - Cipher Feedback\n"
          "4. OFB - Output Feedback\n"
          "5. CTR - Counter mode")
    aes_mode = input()
    while '1' not in aes_mode and '2' not in aes_mode and '3' not in aes_mode and \
          '4' not in aes_mode and '5' not in aes_mode:
        print("Ввод некорректен. Введите номер способа")
        aes_mode = input()

    print("Выберите тип операции, введя её номер:\n"
          "1. Шифрование\n"
          "2. Расшифрование")
    operation = input()
    while '1' not in operation and '2' not in operation:
        print("Ввод некорректен. Введите номер типа операции")
        operation = input()

    print("Выберите режим ввода текста:\n"
          "1. Строка\n"
          "2. Байтовый")
    inp_mode = input()
    while '1' not in operation and '2' not in operation:
        print("Ввод некорректен. Введите номер режима ввода текста")
        inp_mode = input()

    if '1' in operation:
        print("Введите открытый текст")
    if '2' in operation:
        print("Введите шифртекст")
    inp_text = []
    if '1' in inp_mode:
        inp_text = data_split(input())
    if '2' in inp_mode:
        inp_text = input()
        inp_text = data_split([inp_text[i:i + 4] for i in range(0, len(inp_text), 4)], 'byte')

    print('Введите ключ шифрования')
    inp_key = input()
    while len(inp_key) != 16 and len(inp_key) != 24 and len(inp_key) != 32:
        print(f"Вы ввели ключ длиной {len(inp_key)*8} бит."
              f"Пожалуйста, введите ключ корректной длины: 128, 192 или 256 бит.")
        inp_key = input()
    inp_key = list(inp_key)
    for i in range(len(inp_key)):
        inp_key[i] = ord(inp_key[i])

    if len(inp_key) == 16:
        print("Используется AES-128")
    elif len(inp_key) == 24:
        print("Используется AES-192")
    elif len(inp_key) == 32:
        print("Используется AES-256")

    inp_iv = ['0']*16
    if '2' in aes_mode or '3' in aes_mode or '4' in aes_mode or '5' in aes_mode:
        print("Введите IV")
        inp_iv = input()
    while len(inp_iv) != 16:
        print(f"Вы ввели IV длиной {len(inp_key)*8} бит."
              f"Пожалуйста, введите строку длины 128 бит.")
        inp_iv = input()
    inp_iv = list(inp_iv)
    for i in range(len(inp_iv)):
        inp_iv[i] = ord(inp_iv[i])

    res = []
    print("Результат операции:")
    if '1' in operation:
        if '1' in aes_mode:
            res = ecb_encrypt(inp_text, inp_key)
        if '2' in aes_mode:
            res = cbc_encrypt(inp_text, inp_key, inp_iv)
        if '3' in aes_mode:
            res = cfb_encrypt(inp_text, inp_key, inp_iv)
        if '4' in aes_mode:
            res = ofb_process(inp_text, inp_key, inp_iv)
        if '5' in aes_mode:
            res = ofb_process(inp_text, inp_key, inp_iv)
    if '2' in operation:
        if '1' in aes_mode:
            res = ecb_decrypt(inp_text, inp_key)
        if '2' in aes_mode:
            res = cbc_decrypt(inp_text, inp_key, inp_iv)
        if '3' in aes_mode:
            res = cfb_decrypt(inp_text, inp_key, inp_iv)
        if '4' in aes_mode:
            res = ofb_process(inp_text, inp_key, inp_iv)
        if '5' in aes_mode:
            res = ofb_process(inp_text, inp_key, inp_iv)

    print(f"UTF-8 представление: {data_readable(res)}")
    print(f"Байтовое представление: {data_bytes(res)}")


if __name__ == '__main__':
    main()
