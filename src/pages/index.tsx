import React from 'react';
import Link from '@docusaurus/Link';
import Start from './start';
import styles from './index.module.css';
import { Redirect } from '@docusaurus/router';

function Homepage() {
    return (
        <header className={styles.pageContainer}>
            <h1 className={styles.flyingText}>Lemo, I want to fly.</h1>
            <Link to="/docs/web/intro" className={styles.startButton}>
                <span>开始</span>
                <Start width={55} height={55} style={{paddingTop: 5, marginLeft: -10}}/>
            </Link>
        </header>
    );
}

export default function Home(): JSX.Element {
    return <Redirect to="//docs/web/intro" />;
}
