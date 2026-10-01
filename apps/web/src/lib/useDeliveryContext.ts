import { useCallback, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import {
  clearDeliveryContext,
  readDeliveryContext,
  writeDeliveryContext,
  type DeliveryContextPatch,
} from "../types/delivery";

export function useDeliveryContext() {
  const location = useLocation();
  const navigate = useNavigate();
  const context = useMemo(
    () => readDeliveryContext(new URLSearchParams(location.search)),
    [location.search]
  );

  const update = useCallback((patch: DeliveryContextPatch) => {
    const params = writeDeliveryContext(new URLSearchParams(location.search), patch);
    navigate({ pathname: location.pathname, search: params.toString() }, { replace: true });
  }, [location.pathname, location.search, navigate]);

  const clear = useCallback(() => {
    const params = clearDeliveryContext(new URLSearchParams(location.search));
    navigate({ pathname: location.pathname, search: params.toString() }, { replace: true });
  }, [location.pathname, location.search, navigate]);

  return { context, update, clear };
}
