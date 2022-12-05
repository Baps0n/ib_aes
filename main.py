import random
import numpy as np

S_BOX = ['0x0b', '0x84', '0xd3', '0x59', '0xef', '0xf3', '0xaf', '0x2f', '0x7b', '0x4e', '0x3e', '0x0f', '0x52', '0xa9',
         '0x47', '0x1e', '0x74', '0x58', '0x80', '0x6b', '0xf2', '0x8e', '0x0e', '0x6f', '0x33', '0x60', '0xe7', '0xb7',
         '0xc9', '0x12', '0xa6', '0xd2', '0xf6', '0xcb', '0x3d', '0x73', '0x16', '0x8a', '0x14', '0x7c', '0x54', '0x04',
         '0xe1', '0x9c', '0xd6', '0x87', '0xd7', '0xbf', '0x64', '0x86', '0xae', '0x71', '0x1d', '0x00', '0xe9', '0x17',
         '0x13', '0x81', '0x67', '0x62', '0x23', '0xf7', '0xb0', '0xcf', '0x02', '0xd9', '0x26', '0xe0', '0xc0', '0xe4',
         '0x57', '0x53', '0x70', '0x2a', '0x15', '0x31', '0x1c', '0x1b', '0xa8', '0xbc', '0xa7', '0x0a', '0xfc', '0x91',
         '0x9d', '0xe5', '0x30', '0xc8', '0xb3', '0x34', '0x03', '0x25', '0x65', '0x1a', '0xad', '0x4c', '0x43', '0x96',
         '0xff', '0xbd', '0x5b', '0x69', '0xdd', '0x8b', '0xa0', '0xf0', '0xd8', '0xb2', '0x11', '0xc5', '0x42', '0x2b',
         '0x20', '0x97', '0x3a', '0x41', '0x9b', '0xbe', '0x51', '0x90', '0x2c', '0xbb', '0x7a', '0x72', '0x5e', '0x94',
         '0x50', '0x7e', '0xb6', '0xec', '0xe3', '0x63', '0x6e', '0xea', '0x83', '0xb5', '0xb1', '0x29', '0xc3', '0x0c',
         '0xc4', '0x92', '0xac', '0x2e', '0x56', '0xa1', '0xa3', '0xc7', '0xc2', '0x66', '0x6a', '0x09', '0x95', '0x5a',
         '0x06', '0xb8', '0xaa', '0xeb', '0xa2', '0x4a', '0x4d', '0x5d', '0x55', '0xd4', '0x4f', '0xd1', '0x98', '0x9a',
         '0x6d', '0x7f', '0xcd', '0x9e', '0xfa', '0xcc', '0x77', '0x85', '0x9f', '0x38', '0xce', '0x32', '0x6c', '0x75',
         '0xf1', '0x8d', '0x18', '0x89', '0x8f', '0xdc', '0x79', '0xee', '0xb9', '0x24', '0x68', '0x76', '0xba', '0x3f',
         '0xc1', '0x22', '0xdb', '0xf8', '0x5c', '0xc6', '0x5f', '0x10', '0x08', '0x1f', '0xdf', '0x49', '0x46', '0x4b',
         '0x93', '0x01', '0x36', '0x3c', '0xfb', '0xa5', '0x2d', '0x28', '0x39', '0x21', '0x35', '0xb4', '0x3b', '0xe6',
         '0xda', '0x37', '0xa4', '0x27', '0x48', '0x78', '0xed', '0x99', '0x05', '0xf4', '0xf5', '0x61', '0x0d', '0xfd',
         '0xe2', '0x45', '0x44', '0x19', '0xca', '0xab', '0x07', '0x40', '0xe8', '0x7d', '0xf9', '0xd5', '0x82', '0xfe',
         '0x8c', '0xd0', '0xde', '0x88']


def s_box_gen():
    s_box = list(range(256))
    for i in range(256):
        s_box[i] = '0x' + format(s_box[i], "02x")
    random.shuffle(s_box)
    return s_box


def inv_s_box_gen(s_box):
    inv_s_box = [''] * 256
    for i in range(256):
        inv_s_box[int(s_box[i], 0)] = '0x' + format(i, "02x")
    return inv_s_box


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


def sub_bytes(state, s_box):
    for i in range(4):
        for j in range(4):
            # print(S_BOX[state[i][j]], int(S_BOX[state[i][j]], 0))
            state[i][j] = int(s_box[state[i][j]], 0)
    return state


def inv_sub_bytes(state, inv_s_box):
    for i in range(4):
        for j in range(4):
            state[i][j] = int(inv_s_box[state[i][j]], 0)
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


def key_expansion(key, s_box):
    nk = len(key) // 4
    rcon = rcon_gen()

    w = np.empty((4, get_nr(nk) * 4 + 4), dtype="int")
    for i in range(4):
        for j in range(nk):
            w[i][j] = key[i + 4 * j]
    if nk == 4:
        for i in range(4, 10 * 4 + 4):
            if i % 4 == 0:
                w_prev = [w[1][i - 1], w[2][i - 1], w[3][i - 1], w[0][i - 1]]
                for j in range(4):
                    w_prev[j] = int(s_box[w_prev[j]], 0)
                    w[j][i] = w[j][i - 4] ^ w_prev[j] ^ rcon[j][i // 4 - 1]
            else:
                for j in range(4):
                    w[j][i] = w[j][i - 4] ^ w[j][i - 1]
    if nk == 6:
        for i in range(6, 12 * 4 + 4):
            if i % 6 == 0:
                w_prev = [w[1][i - 1], w[2][i - 1], w[3][i - 1], w[0][i - 1]]
                for j in range(4):
                    w_prev[j] = int(s_box[w_prev[j]], 0)
                    w[j][i] = w[j][i - 6] ^ w_prev[j] ^ rcon[j][i // 6 - 1]
            else:
                for j in range(4):
                    w[j][i] = w[j][i - 6] ^ w[j][i - 1]
    if nk == 8:
        for i in range(8, 14 * 4 + 4):
            if i % 8 == 0:
                w_prev = [w[1][i - 1], w[2][i - 1], w[3][i - 1], w[0][i - 1]]
                for j in range(4):
                    w_prev[j] = int(s_box[w_prev[j]], 0)
                    w[j][i] = w[j][i - 8] ^ w_prev[j] ^ rcon[j][i // 8 - 1]
            elif (i - 4) % 8 == 0:
                for j in range(4):
                    w[j][i] = w[j][i - 8] ^ int(s_box[w[j][i - 1]], 0)
            else:
                for j in range(4):
                    w[j][i] = w[j][i - 8] ^ w[j][i - 1]
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
    # print(key)
    return key


def aes_encrypt(inp, key, s_box):
    nk = len(key) // 4
    keys = key_expansion(key, s_box)
    state = fill_state(inp)
    nr = get_nr(nk)

    state = add_round_key(state, get_key_block(keys, 0))
    for i in range(1, nr):
        state = sub_bytes(state, s_box)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, get_key_block(keys, i))

    state = sub_bytes(state, s_box)
    state = shift_rows(state)
    state = add_round_key(state, get_key_block(keys, nr))

    res = list(range(16))
    for i in range(4):
        for j in range(4):
            res[i + 4 * j] = state[i][j]
    return res


def aes_decrypt(inp, key, s_box):
    nk = len(key) // 4
    keys = key_expansion(key, s_box)
    state = fill_state(inp)
    nr = get_nr(nk)
    inv_s_box = inv_s_box_gen(s_box)

    state = add_round_key(state, get_key_block(keys, nr))
    for i in reversed(range(1, nr)):
        state = inv_sub_bytes(state, inv_s_box)
        state = inv_shift_rows(state)
        state = add_round_key(state, get_key_block(keys, i))
        state = inv_mix_columns(state)

    state = inv_shift_rows(state)
    state = inv_sub_bytes(state, s_box)
    state = add_round_key(state, get_key_block(keys, 0))

    res = list(range(16))
    for i in range(4):
        for j in range(4):
            res[i + 4 * j] = state[i][j]
    return res


def main():
    key = [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c]
    sim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sim_24 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    sim_32 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31]
    # print(s_box_gen())
    # print(fill_state(sim))

    # print(inv_shift_rows(shift_rows(fill_state(sim))))
    # print(inv_sub_bytes(sub_bytes(fill_state(sim))))
    # print(inv_mix_columns(mix_columns(fill_state(sim))))
    # print(add_round_key(add_round_key(fill_state(sim), fill_state(key)), fill_state(key)))

    # print(rcon_gen())
    # print(inv_s_box_gen(S_BOX))
    enc = aes_encrypt(sim, sim_32, S_BOX)
    print(sim)
    print(enc)
    print(aes_decrypt(enc, sim_32, S_BOX))


if __name__ == '__main__':
    main()
