export const getApiBaseUrl = () => {
  return typeof window === "undefined"
    ? process.env.API_INTERNAL_URL || "http://server:8000" // for SSR
    : process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"; // for browser
};
