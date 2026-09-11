# Guía de publicación del portafolio en GitHub Pages

## 1. Crear el repositorio
1. Inicia sesión en GitHub.
2. Crea un repositorio público, por ejemplo: `portafolio-digital`.
3. No agregues README automáticamente si vas a subir esta carpeta completa.

## 2. Subir el contenido
Desde una terminal ubicada dentro de la carpeta `portfolio-jdv`:

```bash
git init
git add .
git commit -m "Publicar portafolio digital"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/portafolio-digital.git
git push -u origin main
```

## 3. Activar GitHub Pages
1. En GitHub entra a **Settings > Pages**.
2. En **Build and deployment**, selecciona **Deploy from a branch**.
3. Selecciona rama `main` y carpeta `/docs`.
4. Guarda los cambios.
5. GitHub mostrará la URL pública del portafolio.

## 4. Completar el perfil antes de entregar
- Reemplazar `AGREGAR_URL_LINKEDIN` por la URL real.
- Reemplazar `AGREGAR_URL_GITHUB` por la URL real.
- Agregar otro perfil profesional si corresponde.
- El avatar con iniciales ya cumple como avatar, pero puede reemplazarse por una fotografía profesional.
- Copiar la URL de GitHub Pages en LinkedIn.
- Entregar al docente el enlace del repositorio y el enlace público del portafolio.

## 5. Evidencia recomendada para la entrega
- Captura de la portada del repositorio.
- Captura del perfil/README.
- Captura de la sección Projects.
- Captura de GitHub Pages publicada.
- URL del repositorio.
- URL del portafolio web.
