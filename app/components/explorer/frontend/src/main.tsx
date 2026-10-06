import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Streamlit, type RenderData } from "streamlit-component-lib";
import App from "./App";
import type { Args } from "./types";
import "./styles.css";

function Root() {
  const [args, setArgs] = useState<Args | null>(null);

  useEffect(() => {
    const onRender = (event: Event) => {
      const detail = (event as CustomEvent<RenderData>).detail;
      setArgs(detail.args as Args);
    };
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
    Streamlit.setComponentReady();
    return () => Streamlit.events.removeEventListener(Streamlit.RENDER_EVENT, onRender);
  }, []);

  return <App args={args} />;
}

createRoot(document.getElementById("root")!).render(<Root />);
