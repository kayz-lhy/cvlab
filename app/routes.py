from flask import Blueprint, jsonify, g
from app.auth.decorators import token_required
from app.utils.response_helper import ResponseHelper

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return ResponseHelper.success('请求成功', {
        'message': 'Flask CVLab API',
        'version': '1.0.0'
    })

@main_bp.route('/protected')
@token_required
def protected():
    return ResponseHelper.success('访问受保护资源成功', {
        'message': '这是受保护的路由',
        'user': g.current_user.to_dict()
    })