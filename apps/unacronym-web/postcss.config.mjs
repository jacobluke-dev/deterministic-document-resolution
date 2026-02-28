export default {
  plugins: {
    tailwindcss: {},
    "postcss-flexbugs-fixes": {},
    "postcss-preset-env": {
      stage: 3,
      features: { "nesting-rules": true },
    },
    autoprefixer: {},
  },
};
