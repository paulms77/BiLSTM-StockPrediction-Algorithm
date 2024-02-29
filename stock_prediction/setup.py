from setuptools import setup, find_packages

setup(
    name='stock_prediction_program',
    version='1.0',
    author='Paul77ms',
    author_email='clash833277@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        'stock_prediction_program': ['*.py'],
        'stock_prediction_program.my_package': ['*.py'],
        'stock_prediction_program.my_path.xgb': ['*.joblib'],
        'stock_prediction_program.my_path.vae': ['*.pt'],
        'stock_prediction_program.my_path.lstm': ['*.pt'],
        'stock_prediction_program.my_data': ['*.csv'],
    },