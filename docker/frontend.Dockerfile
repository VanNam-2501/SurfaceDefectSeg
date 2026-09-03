FROM node:22.13.1-bookworm-slim

ENV NODE_ENV=development

WORKDIR /app

COPY apps/web_demo/package.json apps/web_demo/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY apps/web_demo/ ./

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]

