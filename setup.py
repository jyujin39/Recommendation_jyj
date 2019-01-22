from setuptools import setup, find_packages

setup(
    name             = 'recommenderjyj',
    version          = '1.0',
    description      = 'movie recommendation module',
    author           = 'Yujin Jeon',
    author_email     = 'jyujin39@gmail.com',
    url              = 'https://github.com/jyujin39/Recommendation_System',
    download_url     = 'https://github.com/jyujin39/Recommendation_System/blob/master/',
    install_requires = ["numpy", "pandas", "scipy" ],
    keywords         = ['recommendation', 'movie'],
    zip_safe=False,
    python_requires  = '>=3',
)
