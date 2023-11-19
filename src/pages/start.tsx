import React from 'react';
import styles from './index.module.css';

interface Props {
    width: number;
    height: number;
    style?: any;
}

export default ({width, height, style}: Props) => {
    return (
        <svg
            
            viewBox="0 0 1024 1024"
            version="1.1"
            p-id="3541"
            id="mx_n_1700414649336"
            style={style}
            width={width}
            height={height}
        >
            <path
                className={styles.icon}
                d="M604.7488 513.659733l-196.292267 196.9024a36.872533 36.872533 0 0 0-1.041066 50.978134 33.8048 33.8048 0 0 0 48.810666 1.019733l218.2144-218.478933c16.648533-16.669867 16.64-43.677867-0.0256-60.330667L455.466667 264.917333a33.493333 33.493333 0 0 0-48.349867 1.041067 36.5312 36.5312 0 0 0 1.041067 50.496l196.590933 197.201067z"
                p-id="3542"
            ></path>
        </svg>
    );
};
