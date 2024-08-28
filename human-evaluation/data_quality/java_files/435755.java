public static byte calc(byte[] data, int len) {
        byte crc = 0;

        for (int i = 0; i < len; i++)
            crc = CRC8_TABLE[(crc ^ data[i]) & 0xff];

        return crc;
    }