### clone repos

PKG=transformers
VERSION=v2.11.0
REPO=/tmp/repo

PKGDIR="/tmp/${PKG}_${VERSION}"

mkdir "${PKGDIR}"

git clone -b "${VERSION}" --depth 1 https://github.com/huggingface/transformers "${PKGDIR}/transformers"

git clone -b python-v0.7.0 --depth 1 https://github.com/huggingface/tokenizers "${PKGDIR}/tokenizers"

git clone https://bitbucket.org/mrabarnett/mrab-regex.git "${PKGDIR}/regex"
cd "${PKGDIR}/regex"
git checkout fe9fb05

git clone https://github.com/alvations/sacremoses "${PKGDIR}/sacremoses"
cd "${PKGDIR}/sacremoses"
git checkout b94654f

git clone -b v0.1.92 --depth 1 https://github.com/google/sentencepiece "${PKGDIR}/sentencepiece"

### remove git files

find "${PKGDIR}" -name ".git" | xargs rm -rf

# get rust installer from https://forge.rust-lang.org/infra/other-installation-methods.html
# tar -xf rust-nightly-powerpc64le-unknown-linux-gnu.tar.gz
cd rust-nightly-powerpc64le-unknown-linux-gnu
./install.sh --prefix=/root/rust
export PATH=$PATH:/root/rust/bin

cd "${PKGDIR}/tokenizers/bindings/python"
cargo fetch

# zip compress /root/.cargo


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

PKG=transformers
VERSION=v2.11.0
PKGDIR="/tmp/${PKG}_${VERSION}"

# get rust installer from https://forge.rust-lang.org/infra/other-installation-methods.html
# tar -xf rust-nightly-powerpc64le-unknown-linux-gnu.tar.gz
cd rust-nightly-powerpc64le-unknown-linux-gnu
./install.sh --prefix=/root/rust
export PATH=$PATH:/root/rust/bin
cp -r "${PKGDIR}/.cargo" ~/

cd "${PKGDIR}/tokenizers/bindings/python"
pip install setuptools_rust
python setup.py bdist_wheel
pip install dist/...

cd "${PKGDIR}/regex"
python setup.py bdist_wheel
pip install dist/...

cd "${PKGDIR}/sacremoses"
python setup.py bdist_wheel
pip install dist/...

cd "${PKGDIR}/sentencepiece"
mkdir build
cd build
cmake ..
make -j 4
mkdir /tmp/sp
make DESTDIR=/tmp/sp install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/sp/usr/local/lib
ldconfig -v
cd "${PKGDIR}/sentencepiece/python"
PKG_CONFIG_PATH=/tmp/sp/usr/local/lib/pkgconfig CPLUS_INCLUDE_PATH=/tmp/sp/usr/local/include LIBRARY_PATH=/tmp/sp/usr/local/lib python setup.py bdist_wheel
pip install dist/...

cd "${PKGDIR}/transformers"
python setup.py bdist_wheel
pip install dist/...


### fix dependencies

mkdir "${PKGDIR}/whl"

pip install auditwheel==3.1.1

find "${PKGDIR}" -name "*.whl" -not -path "${PKGDIR}/whl/*" -print0 | while read -d $'\0' file
do
  echo "Processing ${file}"
  auditwheel repair -w "${PKGDIR}/whl" "${file}"
done
