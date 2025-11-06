#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * 检查字符串是否需要在 YAML 中加引号
 */
function needsQuotes(str) {
  // YAML 特殊字符
  const specialChars = /[:\?\-\[\]\{\}\#\&\*\!\|\>\'\"\%\@\`]/;

  // 如果以特殊字符开头，或包含冒号、问号等，需要引号
  if (specialChars.test(str)) {
    return true;
  }

  // 如果已经有引号了，不需要再加
  if ((str.startsWith('"') && str.endsWith('"')) ||
      (str.startsWith("'") && str.endsWith("'"))) {
    return false;
  }

  return false;
}

/**
 * 给字符串添加引号（如果需要的话）
 */
function addQuotesIfNeeded(str) {
  if (!str) return str;

  // 已经有引号了
  if ((str.startsWith('"') && str.endsWith('"')) ||
      (str.startsWith("'") && str.endsWith("'"))) {
    return str;
  }

  if (needsQuotes(str)) {
    // 使用双引号，并转义内部的双引号
    return `"${str.replace(/"/g, '\\"')}"`;
  }

  return str;
}

/**
 * 修复文件的 frontmatter
 */
function fixFrontmatter(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');

    // 检查是否有 frontmatter
    if (!content.trimStart().startsWith('---')) {
      return false;
    }

    const lines = content.split('\n');
    let inFrontmatter = false;
    let frontmatterEndIndex = -1;
    let modified = false;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      if (line === '---') {
        if (!inFrontmatter) {
          inFrontmatter = true;
        } else {
          frontmatterEndIndex = i;
          break;
        }
        continue;
      }

      if (inFrontmatter) {
        // 检查 title 和 sidebar_label
        const titleMatch = line.match(/^(title|sidebar_label):\s*(.+)$/);
        if (titleMatch) {
          const key = titleMatch[1];
          const value = titleMatch[2].trim();
          const quotedValue = addQuotesIfNeeded(value);

          if (quotedValue !== value) {
            lines[i] = `${key}: ${quotedValue}`;
            modified = true;
          }
        }
      }
    }

    if (modified) {
      fs.writeFileSync(filePath, lines.join('\n'), 'utf-8');
      console.log(`✅ 已修复 ${filePath}`);
      return true;
    } else {
      console.log(`⏭️  跳过 ${filePath} (无需修改)`);
      return false;
    }
  } catch (error) {
    console.error(`❌ 处理 ${filePath} 失败:`, error.message);
    return false;
  }
}

/**
 * 批量处理文件
 */
function processFiles(files) {
  let fixed = 0;
  let skipped = 0;
  let failed = 0;

  console.log(`\n开始处理 ${files.length} 个文件...\n`);

  for (const file of files) {
    const result = fixFrontmatter(file);
    if (result === true) {
      fixed++;
    } else if (result === false) {
      skipped++;
    } else {
      failed++;
    }
  }

  console.log(`\n处理完成:`);
  console.log(`  ✅ 已修复: ${fixed}`);
  console.log(`  ⏭️  已跳过: ${skipped}`);
  console.log(`  ❌ 失败: ${failed}`);

  return { fixed, skipped, failed };
}

// 主函数
function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.error('用法: node fix-frontmatter-quotes.js <文件1> [文件2] [文件3] ...');
    console.error('或者: find docs -name "*.md" | xargs node fix-frontmatter-quotes.js');
    process.exit(1);
  }

  processFiles(args);
}

// 如果直接运行此脚本
if (require.main === module) {
  main();
}

module.exports = { fixFrontmatter, processFiles };
