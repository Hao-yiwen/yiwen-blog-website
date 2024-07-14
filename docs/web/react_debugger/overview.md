# react工作过程

## 介绍

react是一个web框架，主要是声明式语法减少开发者成本，使用虚拟dom和事件切片优先级调度来更好的解决web应用渲染卡顿和提升性能。

## 工作流程

react通过babel插件来将jsx转化为React.createElement语法，简化开发者开发成本。然后在scheduler中的使用时间切片做到的优先级任务调度，该任务调度直到任务队列中没有任务才会执行完毕。该任务调度会调度reconciler来维护一颗虚拟dom树。该树是由fiber节点使用链表构成的，fiber在react16推出。该虚拟dom树是当前页面状态的真实写照。然后根据diff算法最小化更新真实dom节点。在提交阶段，会真实出发dom更新和触发各个时期的声明周期函数和hooks。然后在事件响应的时候，在raect-dom库中有事件监听事件，所有事件都会被标准化并以合成事件的形式处理。事件处理函数会根据事件的优先级被调度执行。然后调用flushsync函数来在微任务中更新虚拟dom节点，然后反馈到真实dom中，然后在提交阶段最小更新dom。

## 为什么说时间切片是基于fiber节点做到的？

因为fiber节点将页面渲染分解成了颗粒度更小的dom节点渲染，然后在循环的时候，加以优先级是很容易实现中断处理更高优先级事项的。所以说fiber架构是事件粉片和优先级调度的核心概念是没有错的。

## 事件分片和优先级调度是那段函数？

```js
function unstable_scheduleCallback(priorityLevel, callback, options) {
    var currentTime = getCurrentTime();

    var startTime;
    if (typeof options === 'object' && options !== null) {
        var delay = options.delay;
        if (typeof delay === 'number' && delay > 0) {
            startTime = currentTime + delay;
        } else {
            startTime = currentTime;
        }
    } else {
        startTime = currentTime;
    }

    // 在这段代码中可以看到所谓的事件切片就是会根据不同优先级来定义过期事件，然后过期时间越短优先级越高。
    var timeout;
    switch (priorityLevel) {
        case ImmediatePriority:
            timeout = IMMEDIATE_PRIORITY_TIMEOUT;
            break;
        case UserBlockingPriority:
            timeout = USER_BLOCKING_PRIORITY_TIMEOUT;
            break;
        case IdlePriority:
            timeout = IDLE_PRIORITY_TIMEOUT;
            break;
        case LowPriority:
            timeout = LOW_PRIORITY_TIMEOUT;
            break;
        case NormalPriority:
        default:
            timeout = NORMAL_PRIORITY_TIMEOUT;
            break;
    }

    var expirationTime = startTime + timeout;

    var newTask = {
        id: taskIdCounter++,
        callback,
        priorityLevel,
        startTime,
        expirationTime,
        sortIndex: -1,
    };
    // 性能测试
    if (enableProfiling) {
        newTask.isQueued = false;
    }

    if (startTime > currentTime) {
        // 对于超时任务
        newTask.sortIndex = startTime;
        push(timerQueue, newTask);
        if (peek(taskQueue) === null && newTask === peek(timerQueue)) {
            // All tasks are delayed, and this is the task with the earliest delay.
            if (isHostTimeoutScheduled) {
                // Cancel an existing timeout.
                cancelHostTimeout();
            } else {
                isHostTimeoutScheduled = true;
            }
            // 从注释可以看到会在是个时间延迟后进行
            //   function requestHostTimeout(callback, ms) {
            //     taskTimeoutID = localSetTimeout(() => {
            //         callback(getCurrentTime());
            //     }, ms);
            //     }
            requestHostTimeout(handleTimeout, startTime - currentTime);
        }
    } else {
        newTask.sortIndex = expirationTime;
        push(taskQueue, newTask);
        if (enableProfiling) {
            markTaskStart(newTask, currentTime);
            newTask.isQueued = true;
        }
        if (!isHostCallbackScheduled && !isPerformingWork) {
            isHostCallbackScheduled = true;
            // 这里会调度任务执行函数，其中workloop会根据时间切片进行调度。过期时间越短越先执行，而优先级调度最后只会转化为过期时间而已。
            requestHostCallback(flushWork);
        }
    }
    // 返回时为了对事件的引用，例如取消任务。
    return newTask;
}
```

## 根据diff算法最小化更新真实dom节点，这段代码在哪里？

### diff算法发生阶段

渲染阶段（Render Phase）

渲染阶段是指 React 计算和生成 Fiber 树的阶段。在这个阶段，React 会对比新旧虚拟 DOM 树，找出变化，并生成新的 Fiber 树。渲染阶段是纯计算阶段，不会对真实 DOM 进行任何修改。

-   JSX 转换为 React 元素： JSX 被转换为 React 元素对象。对于 Portal 元素，这个过程由 ReactDOM.createPortal 完成。
-   协调（Reconciliation）：在渲染阶段，React 使用协调算法（Diff 算法）来比较新旧虚拟 DOM 树，生成最小的变更集。这些变更集是由 Fiber 节点及其标志（flags）表示的。
-   根据协调结果，使用react-dom库中api增删改dom节点。（此处只是增删改dom节点，但是实际的应用是在提交阶段。）

### 代码所在位置

该文件中的`react/packages/react-reconciler/src/ReactChildFiber.old.js`中的`ChildReconciler`函数是diff算法的核心实现，其中`reconcileChildFibers`函数是核心diff函数。

```js
// 该 API 将使用协调本身的副作用标记子元素。当我们遍历子元素和父元素时，它们将被添加到副作用列表中。
function reconcileChildFibers(
    returnFiber: Fiber,
    currentFirstChild: Fiber | null,
    newChild: any,
    lanes: Lanes,
  ): Fiber | null {
    // 此函数不是递归的。,如果顶层项是一个数组，我们将其视为一组子项，,而不是片段。另一方面，嵌套数组将被视为,片段节点。递归发生在正常流程中。
    const isUnkeyedTopLevelFragment =
      typeof newChild === 'object' &&
      newChild !== null &&
      newChild.type === REACT_FRAGMENT_TYPE &&
      newChild.key === null;
    if (isUnkeyedTopLevelFragment) {
      newChild = newChild.props.children;
    }

    // Handle object types
    if (typeof newChild === 'object' && newChild !== null) {
      // 此处的$$typeof是在jsx转化为fiber节点的时候生成的
      switch (newChild.$$typeof) {
        case REACT_ELEMENT_TYPE:
          return placeSingleChild(
            reconcileSingleElement(
              returnFiber,
              currentFirstChild,
              newChild,
              lanes,
            ),
          );
        case REACT_PORTAL_TYPE:
          return placeSingleChild(
            reconcileSinglePortal(
              returnFiber,
              currentFirstChild,
              newChild,
              lanes,
            ),
          );
        case REACT_LAZY_TYPE:
          const payload = newChild._payload;
          const init = newChild._init;
          // TODO: This function is supposed to be non-recursive.
          return reconcileChildFibers(
            returnFiber,
            currentFirstChild,
            init(payload),
            lanes,
          );
      }

      if (isArray(newChild)) {
        return reconcileChildrenArray(
          returnFiber,
          currentFirstChild,
          newChild,
          lanes,
        );
      }

      if (getIteratorFn(newChild)) {
        return reconcileChildrenIterator(
          returnFiber,
          currentFirstChild,
          newChild,
          lanes,
        );
      }

      throwOnInvalidObjectType(returnFiber, newChild);
    }

    if (
      (typeof newChild === 'string' && newChild !== '') ||
      typeof newChild === 'number'
    ) {
      return placeSingleChild(
        reconcileSingleTextNode(
          returnFiber,
          currentFirstChild,
          '' + newChild,
          lanes,
        ),
      );
    }

    if (__DEV__) {
      if (typeof newChild === 'function') {
        warnOnFunctionType(returnFiber);
      }
    }

    // 其余情况均视为空。
    return deleteRemainingChildren(returnFiber, currentFirstChild);
  }
```

## reconciler中渲染阶段根据diff结果创建statenode是怎么做到的？

```
function performUnitOfWork(unitOfWork: Fiber): void {
  // 这根纤维的当前状态是刷新的。理想情况下，不应该依赖于它，但在这里依赖它意味着我们不需要在进行中的工作上添加额外字段
  const current = unitOfWork.alternate;
  setCurrentDebugFiberInDEV(unitOfWork);

  let next;
  if (enableProfilerTimer && (unitOfWork.mode & ProfileMode) !== NoMode) {
    startProfilerTimer(unitOfWork);
    // diff算法实际执行地方
    next = beginWork(current, unitOfWork, renderLanes);
    stopProfilerTimerIfRunningAndRecordDelta(unitOfWork, true);
  } else {
    // diff算法实际执行地方
    next = beginWork(current, unitOfWork, renderLanes);
  }

  resetCurrentDebugFiberInDEV();
  unitOfWork.memoizedProps = unitOfWork.pendingProps;
  if (next === null) {
    // 渲染阶段实际创建statenode的地方
    completeUnitOfWork(unitOfWork);
  } else {
    workInProgress = next;
  }

  ReactCurrentOwner.current = null;
}
```

从这里可以简单看到，实际上diff算法是直接修改传递给他对对象的参数，然后completeUnitOfWork根据修改后的fiber节点来创建dom。。。虽然在开发中直接修改对象参数不太好。但是react就是这样做的～～～这也是侧面证明了js的灵活性吧。。～～～～

当然从这里也可以看到fiber变化是没有规律的，所以在react并不对外直接开发fiber节点和依赖fiber节点干任何事情。而fiber节点只是内部用的～～还有就是使用react devtools调试时候用的。

## 该事件根据事件的优先级，进行事件响应，代码怎么实现的？

## Mutation Phase中的commitMutationEffect函数是如何将变更应用到真实dom上的？

改函数是实际的提交fiber节点到dom真实节点。其中commitUpdate是实际的dom刷新函数，在rn和web各不相同。（所有类组建或者函数组件等其他组件最终都会被便利为实际的hostcomponent，然后使用commitupdate来刷新dom真实节点。）

可以看到该函数中很多地方都是直接使用statenode，而statenode就是在渲染阶段创建的真实dom，只是在提交阶段一起提交而已。

```js
function commitMutationEffectsOnFiber(
  finishedWork: Fiber,
  root: FiberRoot,
  lanes: Lanes,
) {
  const current = finishedWork.alternate;
  const flags = finishedWork.flags;
  switch (finishedWork.tag) {
    case FunctionComponent:
    case ForwardRef:
    case MemoComponent:
    case SimpleMemoComponent: {
      recursivelyTraverseMutationEffects(root, finishedWork, lanes);
      commitReconciliationEffects(finishedWork);

      if (flags & Update) {
        try {
          commitHookEffectListUnmount(
            HookInsertion | HookHasEffect,
            finishedWork,
            finishedWork.return,
          );
          commitHookEffectListMount(
            HookInsertion | HookHasEffect,
            finishedWork,
          );
        } catch (error) {
          captureCommitPhaseError(finishedWork, finishedWork.return, error);
        }
        if (
          enableProfilerTimer &&
          enableProfilerCommitHooks &&
          finishedWork.mode & ProfileMode
        ) {
          try {
            startLayoutEffectTimer();
            commitHookEffectListUnmount(
              HookLayout | HookHasEffect,
              finishedWork,
              finishedWork.return,
            );
          } catch (error) {
            captureCommitPhaseError(finishedWork, finishedWork.return, error);
          }
          recordLayoutEffectDuration(finishedWork);
        } else {
          try {
            commitHookEffectListUnmount(
              HookLayout | HookHasEffect,
              finishedWork,
              finishedWork.return,
            );
          } catch (error) {
            captureCommitPhaseError(finishedWork, finishedWork.return, error);
          }
        }
      }
      return;
    }
    case ClassComponent: {
      recursivelyTraverseMutationEffects(root, finishedWork, lanes);
      commitReconciliationEffects(finishedWork);

      if (flags & Ref) {
        if (current !== null) {
          safelyDetachRef(current, current.return);
        }
      }

      if (flags & Callback && offscreenSubtreeIsHidden) {
        const updateQueue: UpdateQueue<
          *,
        > | null = (finishedWork.updateQueue: any);
        if (updateQueue !== null) {
          deferHiddenCallbacks(updateQueue);
        }
      }
      return;
    }
    case HostComponent: {
      recursivelyTraverseMutationEffects(root, finishedWork, lanes);
      commitReconciliationEffects(finishedWork);

      if (flags & Ref) {
        if (current !== null) {
          safelyDetachRef(current, current.return);
        }
      }
      if (supportsMutation) {
        if (finishedWork.flags & ContentReset) {
          const instance: Instance = finishedWork.stateNode;
          try {
            resetTextContent(instance);
          } catch (error) {
            captureCommitPhaseError(finishedWork, finishedWork.return, error);
          }
        }

        if (flags & Update) {
          const instance: Instance = finishedWork.stateNode;
          if (instance != null) {
            // Commit the work prepared earlier.
            const newProps = finishedWork.memoizedProps;
            // For hydration we reuse the update path but we treat the oldProps
            // as the newProps. The updatePayload will contain the real change in
            // this case.
            const oldProps =
              current !== null ? current.memoizedProps : newProps;
            const type = finishedWork.type;
            // TODO: Type the updateQueue to be specific to host components.
            const updatePayload: null | UpdatePayload = (finishedWork.updateQueue: any);
            finishedWork.updateQueue = null;
            if (updatePayload !== null) {
              try {
                commitUpdate(
                  instance,
                  updatePayload,
                  type,
                  oldProps,
                  newProps,
                  finishedWork,
                );
              } catch (error) {
                captureCommitPhaseError(
                  finishedWork,
                  finishedWork.return,
                  error,
                );
              }
            }
          }
        }
      }
      return;
    }
    // ...省略其他代码
}
```
