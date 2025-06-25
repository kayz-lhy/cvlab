#!/bin/bash

# 基础Flask项目目录结构生成脚本
# 使用方法: bash create_flask_project.sh [项目名称]

PROJECT_NAME=${1:-flask_cvlab}

echo "正在创建基础Flask项目: $PROJECT_NAME"

# 创建主项目目录
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# 创建基础应用程序结构

# 创建主要文件（空文件）
touch app/__init__.py
touch app/routes.py
touch app/models.py

# 创建项目根文件
touch config.py
touch run.py
touch requirements.txt
touch .env
touch .gitignore
touch README.md

echo "✅ 基础Flask项目结构创建完成！"
echo ""
echo "📁 项目目录结构:"
echo "$PROJECT_NAME/"
echo "├── app/"
echo "│   ├── __init__.py"
echo "│   ├── routes.py"
echo "│   ├── models.py"
echo "│   ├── static/"
echo "│   │   ├── css/"
echo "│   │   ├── js/"
echo "│   │   └── images/"
echo "│   └── templates/"
echo "├── config.py"
echo "├── run.py"
echo "├── requirements.txt"
echo "├── .env"
echo "├── .gitignore"
echo "└── README.md"
echo ""
echo "🚀 下一步:"
echo "1. cd $PROJECT_NAME"
echo "2. 创建虚拟环境: python -m venv venv"
echo "3. 激活虚拟环境: source venv/bin/activate (Linux/Mac) 或 venv\\Scripts\\activate (Windows)"
echo "4. 安装Flask: pip install flask"
echo "5. 开始开发你的Flask应用！"