from flask import jsonify

class ResponseHelper:
    """统一响应格式化工具"""

    @staticmethod
    def success(msg="请求成功", data=None, code=200):
        """成功响应"""
        return jsonify({
            'code': code,
            'msg': msg,
            'data': data
        }), code

    @staticmethod
    def error(msg="请求失败", code=400, data=None):
        """错误响应"""
        return jsonify({
            'code': code,
            'msg': msg,
            'data': data
        }), code

    @staticmethod
    def unauthorized(msg="未授权访问"):
        """未授权响应"""
        return ResponseHelper.error(msg, 401)

    @staticmethod
    def forbidden(msg="禁止访问"):
        """禁止访问响应"""
        return ResponseHelper.error(msg, 403)

    @staticmethod
    def not_found(msg="资源不存在"):
        """资源不存在响应"""
        return ResponseHelper.error(msg, 404)

    @staticmethod
    def conflict(msg="资源冲突"):
        """资源冲突响应"""
        return ResponseHelper.error(msg, 409)

    @staticmethod
    def server_error(msg="服务器内部错误"):
        """服务器错误响应"""
        return ResponseHelper.error(msg, 500)