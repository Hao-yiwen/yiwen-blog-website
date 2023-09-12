import React from 'react';
import Giscus from '@giscus/react';

export const Comment = () => {
    return (
        <div style={{ paddingTop: 50 }}>
            <Giscus
                id="comments"
                // highlight-warn-start
                // 这部分填写你自己的
                repo="Hao-yiwen/yiwen-blog-website"
                repoId="R_kgDOKCp9Gw"
                category="Announcements"
                categoryId="DIC_kwDOKCp9G84CZP8J"
                // highlight-warn-end
                mapping="pathname"
                strict="0"
                term="Welcome to @giscus/react component!"
                reactionsEnabled="1"
                emitMetadata="0"
                inputPosition="bottom"
                theme="preferred_color_scheme"
                lang="zh-CN"
                loading="lazy"
            />
        </div>
    );
};

export default Comment;
