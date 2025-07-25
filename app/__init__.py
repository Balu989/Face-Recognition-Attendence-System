from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Create db and login_manager objects
db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.secret_key = 'your_secret_key_here'

    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Balutharani%401234@localhost/face_recognition'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'routes.login'

    # Local import to avoid circular import issues
    from .models import User
    from .routes import routes
    app.register_blueprint(routes)

    with app.app_context():
        db.create_all()

    # Flask-Login user_loader
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    return app
