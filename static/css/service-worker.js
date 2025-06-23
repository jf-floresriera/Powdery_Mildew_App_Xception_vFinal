const CACHE_NAME = 'mildiu-cache-v1';
const urlsToCache = [
  '/',
  '/static/css/style.css',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  '/manifest.json'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(urlsToCache);
    })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});
