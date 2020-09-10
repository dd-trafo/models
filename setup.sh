### clone repos

PKG=spacy
VERSION=v2.3.2
REPO=/tmp/repo
PKGDIR="/tmp/${PKG}_${VERSION}"

mkdir "${PKGDIR}"

git clone -b v0.4.2 --depth 1 https://github.com/explosion/cython-blis "${PKGDIR}/cython-blis"
git clone -b v2.0.3 --depth 1 https://github.com/explosion/cymem "${PKGDIR}/cymem"
git clone -b v1.0.2 --depth 1 https://github.com/explosion/murmurhash "${PKGDIR}/murmurhash"
git clone -b v1.19.1 --depth 1 https://github.com/numpy/numpy "${PKGDIR}/numpy"
git clone -b v3.0.2 --depth 1 https://github.com/explosion/preshed "${PKGDIR}/preshed"
git clone -b v0.16.0 --depth 1 https://github.com/tobgu/pyrsistent "${PKGDIR}/pyrsistent"
git clone -b v2.3.2 --depth 1 https://github.com/explosion/spaCy "${PKGDIR}/spaCy"
git clone -b v1.0.2 --depth 1 https://github.com/explosion/srsly "${PKGDIR}/srsly"
git clone -b v7.4.1 --depth 1 https://github.com/explosion/thinc "${PKGDIR}/thinc"

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

### compile wheels

PKG=spacy
VERSION=v2.3.2
PKGDIR="/tmp/${PKG}_${VERSION}"

pip install cython==0.29.21

cd "${PKGDIR}/numpy"
python setup.py bdist_wheel
pip install dist/*.whl

cd "${PKGDIR}/cymem"
python setup.py bdist_wheel
pip install dist/*.whl

cd "${PKGDIR}/murmurhash"
python setup.py bdist_wheel
pip install dist/*.whl

cd "${PKGDIR}/preshed"
python setup.py bdist_wheel
pip install dist/*.whl

cd "${PKGDIR}/srsly"
python setup.py bdist_wheel
pip install dist/*.whl

cd "${PKGDIR}/cython-blis"
BLIS_ARCH="generic" python setup.py bdist_wheel
pip install dist/*.whl

cd "${PKGDIR}/thinc"
python setup.py bdist_wheel
pip install dist/*.whl

cd "${PKGDIR}/pyrsistent"
python setup.py bdist_wheel
pip install dist/*.whl

cd "${PKGDIR}/spaCy"

### replace spacy's x86 numpy with the system version
rm -r "${PKGDIR}/spaCy/include/numpy/"
ln -s /opt/python/.../lib/python3.6/site-packages/numpy/core/include/numpy "${PKGDIR}/spaCy/include/"

python setup.py bdist_wheel
pip install dist/*.whl

### fix dependencies

mkdir "${PKGDIR}/whl"

pip install auditwheel==3.1.1

find "${PKGDIR}" -name "*.whl" -not -path "${PKGDIR}/whl/*" -print0 | while read -d $'\0' file
do
  echo "Processing ${file}"
  auditwheel repair -w "${PKGDIR}/whl" "${file}"
done
