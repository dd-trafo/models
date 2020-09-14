### clone repos

PKG=pytorch
VERSION=v1.6.0
REPO=/tmp/repo

PKGDIR="/tmp/${PKG}_${VERSION}"

mkdir "${PKGDIR}"

git clone -b "${VERSION}" --depth 1 https://github.com/pytorch/pytorch "${PKGDIR}/pytorch"

cd "${PKGDIR}/pytorch"

git submodule sync
git submodule update --init --recursive

### remove git files

find "${PKGDIR}" -name ".git" | xargs rm -rf

exit

### zip

git init "${REPO}"

cd /tmp

zip -s 95m -0 -r "${REPO}/${PKG}_${VERSION}.zip" "${PKG}_${VERSION}/"

### upload


### unzip

zip -FF "${PKG}_${VERSION}.zip" --out target-full.zip

unzip target-full.zip

### compile

PKG=pytorch
VERSION=v1.6.0
PKGDIR="/tmp/${PKG}_${VERSION}"

# get and compile sleef
git clone https://github.com/shibatch/sleef
cd sleef
mkdir build && cd build
cmake ..
make
make DESTDIR=/tmp/sp install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/sp/usr/local/lib
ldconfig -v

# compile torch with pre-compiled sleef
cd "${PKGDIR}/pytorch"

USE_SYSTEM_SLEEF=ON PKG_CONFIG_PATH=/tmp/sp/usr/local/lib/pkgconfig CPLUS_INCLUDE_PATH=/tmp/sp/usr/local/include LIBRARY_PATH=/tmp/sp/usr/local/lib python setup.py bdist_wheel

# maybe those flags aren't needed
#USE_MKLDNN=0 USE_CUDNN=0 USE_CUDA=0 python setup.py bdist_wheel

### fix dependencies

pip install auditwheel==3.1.1

auditwheel repair -w dist/ dist/...whl
