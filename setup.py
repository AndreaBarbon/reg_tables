import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reg_tables",                     
    version="0.0.2",                     
    author="Andrea Barbon, Kirill Kazakov",                     
    author_email='',
    url = 'https://github.com/AndreaBarbon/reg_tables',
    description="",
    license='MIT',
    long_description=long_description,      
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    
    keywords = ['Linear models', 'regression table'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                    
    python_requires='>=3.8',                
    py_modules=["reg_tables"],            
    install_requires=['numpy',
    'pandas',
    'linearmodels']
)
