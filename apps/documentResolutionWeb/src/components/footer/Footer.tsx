// src/components/Footer/Footer.tsx
import Link from "next/link";
import {getApiBaseUrl} from "@/utils/getApiBaseUrl";


type Page = {
  slug: string;
  label: string;
};

const Footer = async () => {
  let data: Page[] = [];

  try {
    const res = await fetch(`${getApiBaseUrl()}/api/static_pages`, {
      next: { revalidate: 3600 },
    });

    if (!res.ok) {
      throw new Error(`Fetch failed: ${res.status} ${res.statusText}`);
    }

    data = await res.json();
    console.error("[Footer] Static pages:", data);
  } catch (error) {
    console.error("[Footer] Fetch error:", error);
  }

  return (
    <footer className="text-center text-sm text-gray-700 mt-10 px-2 py-3 border-t">
      <div className="flex justify-center gap-4">
        {Array.isArray(data) &&
          data.map((page) => (
            <Link
              key={page.slug.trim()}
              href={`/info/${page.slug.trim()}`}
              className="hover:underline transition duration-150"
            >
              {page.label.trim()}
            </Link>
          ))}
      </div>
      <div className="text-xs text-gray-700 mt-4">
        InstructorCompare is a product of{" "}
        <span className="font-semibold">Intellect AI LTD</span>.
      </div>
      <div className="mt-1">© 2025 InstructorCompare · Made in Leeds, UK</div>
    </footer>
  );
};

export default Footer;
