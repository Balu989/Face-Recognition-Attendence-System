# from flask import Flask
# from app.routes import routes
# import os

# # Create Flask app with correct folder paths
# app = Flask(
#     __name__,
#     template_folder=os.path.join(os.path.dirname(__file__), 'app', 'templates'),
#     static_folder=os.path.join(os.path.dirname(__file__), 'app', 'static')
# )

# # Register the routes blueprint
# app.register_blueprint(routes)

# if __name__ == "__main__":
#     app.run(debug=True)




from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
