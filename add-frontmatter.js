#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * 从 git log 获取文件的创建时间（第一次提交）
 */
function getFileCreationDate(filePath) {
  try {
    const result = execSync(
      `git log --follow --format=%aI --reverse -- "${filePath}" | head -1`,
      { encoding: 'utf-8' }
    ).trim();
    return result || new Date().toISOString();
  } catch (error) {
    console.warn(`无法获取 ${filePath} 的创建时间，使用当前时间`);
    return new Date().toISOString();
  }
}

/**
 * 从 git log 获取文件的最后修改时间
 */
function getFileLastModifiedDate(filePath) {
  try {
    const result = execSync(
      `git log -1 --format=%aI -- "${filePath}"`,
      { encoding: 'utf-8' }
    ).trim();
    return result || new Date().toISOString();
  } catch (error) {
    console.warn(`无法获取 ${filePath} 的修改时间，使用当前时间`);
    return new Date().toISOString();
  }
}

/**
 * 从文件内容中提取第一个标题
 */
function extractTitle(content, filePath) {
  // 跳过 frontmatter（如果存在）
  let lines = content.split('\n');
  let startIndex = 0;

  if (lines[0] === '---') {
    const endIndex = lines.findIndex((line, idx) => idx > 0 && line === '---');
    if (endIndex > 0) {
      startIndex = endIndex + 1;
    }
  }

  // 查找第一个 # 标题
  for (let i = startIndex; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('# ')) {
      return line.substring(2).trim();
    }
  }

  // 如果没找到，使用文件名
  return path.basename(filePath, path.extname(filePath))
    .replace(/_/g, ' ')
    .replace(/-/g, ' ');
}

/**
 * 检查文件是否已有 frontmatter
 */
function hasFrontmatter(content) {
  return content.trimStart().startsWith('---');
}

/**
 * 为文件添加 frontmatter
 */
function addFrontmatter(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');

    // 如果已经有 frontmatter，跳过
    if (hasFrontmatter(content)) {
      console.log(`⏭️  跳过 ${filePath} (已有 frontmatter)`);
      return false;
    }

    // 提取标题
    const title = extractTitle(content, filePath);

    // 获取时间戳
    const createdAt = getFileCreationDate(filePath);
    const updatedAt = getFileLastModifiedDate(filePath);

    // 生成 frontmatter
    const frontmatter = `---
title: ${title}
sidebar_label: ${title}
date: ${createdAt.split('T')[0]}
last_update:
  date: ${updatedAt.split('T')[0]}
---

`;

    // 写入新内容
    const newContent = frontmatter + content;
    fs.writeFileSync(filePath, newContent, 'utf-8');

    console.log(`✅ 已处理 ${filePath}`);
    return true;
  } catch (error) {
    console.error(`❌ 处理 ${filePath} 失败:`, error.message);
    return false;
  }
}

/**
 * 批量处理文件
 */
function processFiles(files) {
  let processed = 0;
  let skipped = 0;
  let failed = 0;

  console.log(`\n开始处理 ${files.length} 个文件...\n`);

  for (const file of files) {
    const result = addFrontmatter(file);
    if (result === true) {
      processed++;
    } else if (result === false) {
      skipped++;
    } else {
      failed++;
    }
  }

  console.log(`\n处理完成:`);
  console.log(`  ✅ 已处理: ${processed}`);
  console.log(`  ⏭️  已跳过: ${skipped}`);
  console.log(`  ❌ 失败: ${failed}`);

  return { processed, skipped, failed };
}

// 主函数
function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.error('用法: node add-frontmatter.js <文件1> [文件2] [文件3] ...');
    process.exit(1);
  }

  processFiles(args);
}

// 如果直接运行此脚本
if (require.main === module) {
  main();
}

module.exports = { addFrontmatter, processFiles };
