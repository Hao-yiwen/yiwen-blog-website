# useStateRef

## useStateRef 的好处
- 即时访问最新状态：使用 useState 时，状态更新可能是异步的，这意味着在状态更新后立即读取状态可能不会反映最新值。而使用 useRef，可以确保在任何时间点都可以访问最新的状态。
- 避免闭包陷阱：在某些情况下，尤其是在回调函数中，闭包可能会捕获旧的状态值。通过使用 useRef，可以确保总是引用到最新的状态。

## 代码实现
```ts
export const useStateRef = <S>(initialState: S | (() => S)): [S, Dispatch<SetStateAction<S>>, MutableRefObject<S>] => {
  const [state, setState] = useState(initialState);
  const stateRef = useRef(state);
  const setStateHandle = useCallback((val: S | ((pre?: S) => S)) => {
    if (typeof val === 'function') {
      // stateRef.current = val();
      // 不怎么做会提示报错
      // Not all constituents of type '(() => S) | (S & Function)' are callable.
      const temp = val as (pre?: S) => S;
      stateRef.current = temp(stateRef.current);
    } else {
      stateRef.current = val;
    }
    setState(val);
  }, []);
  return [state, setStateHandle as Dispatch<SetStateAction<S>>, stateRef];
};
```