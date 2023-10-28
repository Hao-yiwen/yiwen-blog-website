import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

import styles from './index.module.css';

function Homepage() {
    return (
        <header className={styles.pageContainer}>
            <div className={styles.pageCenter}>
                <div className={styles.pageFont}>Lemo, I want to fly.</div>
                <Link to="/docs/web/intro" className={styles.link}>
                    天大寒 硕冰坚
                </Link>
            </div>
        </header>
    );
}

export default function Home(): JSX.Element {
    return (
        <div className={styles.container}>
            <Homepage />
        </div>
    );
}
